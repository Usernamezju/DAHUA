"""Build an NTU25/120-class incremental distillation annotation pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from dahua_cup.pipeline.collect_ntu120_pseudo import load_labels
from dahua_cup.pipeline.common import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-annotation", required=True)
    parser.add_argument("--pseudo-jsonl", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-replay-samples", type=int, default=4000)
    parser.add_argument("--max-pseudo-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(annotation: dict) -> str:
    value = annotation.get("frame_dir") or annotation.get("filename")
    if not value:
        raise ValueError("base annotation has no frame_dir or filename")
    return str(value)


def _stratified_replay(
    annotations: Sequence[dict],
    allowed_ids: set[str],
    capacity: int,
    seed: int,
) -> list[dict]:
    candidates = [
        annotation
        for annotation in annotations
        if _identifier(annotation) in allowed_ids
    ]
    if capacity <= 0 or len(candidates) <= capacity:
        return candidates
    by_class = defaultdict(list)
    for annotation in candidates:
        by_class[int(annotation["label"])].append(annotation)
    rng = random.Random(seed)
    for values in by_class.values():
        values.sort(key=_identifier)
        rng.shuffle(values)
    selected = []
    classes = sorted(by_class)
    cursor = 0
    while len(selected) < capacity and classes:
        label = classes[cursor % len(classes)]
        values = by_class[label]
        if values:
            selected.append(values.pop())
        if not values:
            classes.remove(label)
            cursor = 0
        else:
            cursor += 1
    return selected


def _distill_fields(
    annotation: dict,
    label_space_size: int,
    *,
    hard: bool,
) -> dict:
    value = dict(annotation)
    value.update(
        {
            "has_hard_label": int(hard),
            "teacher_distribution": np.zeros(
                label_space_size, dtype=np.float32
            ),
            "teacher_valid": 0,
            "quality_weight": 1.0,
            "previous_distribution": np.zeros(
                label_space_size, dtype=np.float32
            ),
            "previous_valid": 0,
            "label_source": "ntu120_replay" if hard else "pseudo",
        }
    )
    return value


def _pseudo_annotation(row: dict, labels: Sequence[str]) -> dict:
    if row.get("status") != "accepted":
        raise ValueError("only accepted NTU120 pseudo labels may enter training")
    vector = np.asarray(row.get("soft_label"), dtype=np.float32)
    if vector.shape != (len(labels),) or not np.isfinite(vector).all():
        raise ValueError("NTU120 pseudo soft_label must contain 120 finite values")
    if np.any(vector < 0) or not np.isclose(vector.sum(), 1.0, atol=1e-5):
        raise ValueError("NTU120 pseudo soft_label must be non-negative and sum to one")
    feature_path = Path(str(row.get("feature_path", ""))).expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"pseudo feature not found: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as artifact:
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)
    if (
        keypoint.ndim != 4
        or keypoint.shape[0] not in (1, 2)
        or keypoint.shape[2:] != (25, 3)
        or keypoint.shape[1] < 1
    ):
        raise ValueError(f"NTU120 pseudo feature has invalid shape: {keypoint.shape}")
    label = str(row["label"])
    if label not in labels:
        raise ValueError(f"pseudo label is outside NTU120: {label}")
    return {
        "frame_dir": f"pseudo_{row['sample_id']}",
        "sample_id": str(row["sample_id"]),
        "keypoint": keypoint,
        "total_frames": int(keypoint.shape[1]),
        "label": labels.index(label),
        "has_hard_label": 0,
        "teacher_distribution": vector,
        "teacher_valid": 1,
        "quality_weight": float(row.get("quality_score", 0.0)),
        "previous_distribution": np.zeros(len(labels), dtype=np.float32),
        "previous_valid": 0,
        "label_source": "qwen_ntu120_pseudo",
        "pseudo_fingerprint": str(row.get("fingerprint", "")),
    }


def build_dataset(
    base_data: dict,
    pseudo_rows: Sequence[dict],
    labels: Sequence[str],
    *,
    max_replay_samples: int = 4000,
    max_pseudo_samples: int = 2000,
    seed: int = 20260729,
) -> tuple[dict, dict]:
    if len(labels) != 120:
        raise ValueError("NTU120 distillation requires exactly 120 labels")
    split = dict(base_data.get("split") or {})
    base_annotations = list(base_data.get("annotations") or [])
    if "xsub_train" not in split or "xsub_val" not in split:
        raise ValueError("base annotation must contain xsub_train and xsub_val")
    try:
        observed_labels = {
            int(annotation["label"]) for annotation in base_annotations
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("base annotation contains an invalid hard label") from exc
    expected_labels = set(range(len(labels)))
    if observed_labels != expected_labels:
        raise ValueError(
            "base annotation is not a complete NTU120 label space; "
            f"observed {len(observed_labels)} classes"
        )
    replay = _stratified_replay(
        base_annotations,
        set(map(str, split["xsub_train"])),
        max_replay_samples,
        seed,
    )
    validation_ids = set(map(str, split["xsub_val"]))
    validation = [
        annotation
        for annotation in base_annotations
        if _identifier(annotation) in validation_ids
    ]
    accepted = sorted(
        (row for row in pseudo_rows if row.get("status") == "accepted"),
        key=lambda row: (-float(row.get("quality_score", 0.0)), str(row["sample_id"])),
    )
    if max_pseudo_samples > 0:
        accepted = accepted[:max_pseudo_samples]
    replay_values = [
        _distill_fields(annotation, len(labels), hard=True)
        for annotation in replay
    ]
    validation_values = [
        _distill_fields(annotation, len(labels), hard=True)
        for annotation in validation
    ]
    pseudo_values = [_pseudo_annotation(row, labels) for row in accepted]
    train_ids = [_identifier(row) for row in replay_values + pseudo_values]
    val_ids = [_identifier(row) for row in validation_values]
    data = {
        "split": {
            "xsub_train": train_ids,
            "xsub_val": val_ids,
        },
        "annotations": replay_values + pseudo_values + validation_values,
    }
    metadata = {
        "schema_version": "ntu120_distill_dataset.v1",
        "labels": list(labels),
        "replay_samples": len(replay_values),
        "pseudo_samples": len(pseudo_values),
        "validation_samples": len(validation_values),
        "pseudo_sample_ids": [row["sample_id"] for row in accepted],
        "pseudo_fingerprints": [
            str(row.get("fingerprint", "")) for row in accepted
        ],
    }
    return data, metadata


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_replay_samples < 0 or args.max_pseudo_samples < 0:
        raise ValueError("sample limits must be non-negative")
    with Path(args.base_annotation).open("rb") as stream:
        base_data = pickle.load(stream)
    labels = list(load_labels(args.label_map))
    data, metadata = build_dataset(
        base_data,
        read_jsonl(args.pseudo_jsonl),
        labels,
        max_replay_samples=args.max_replay_samples,
        max_pseudo_samples=args.max_pseudo_samples,
        seed=args.seed,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    metadata["annotation_sha256"] = _sha256(destination)
    metadata_path = destination.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
