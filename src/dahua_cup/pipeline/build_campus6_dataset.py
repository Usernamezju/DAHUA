"""Build a versioned NTU25 ProtoGCN annotation file with pseudo/replay signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.common import log_event, read_jsonl
from dahua_cup.semantic_teacher.schemas import LABELS, normalize_distribution


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
HARD_SOURCES = frozenset(("human", "manual", "human_review", "replay"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--pseudo-jsonl")
    parser.add_argument("--replay-jsonl")
    parser.add_argument("--previous-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--parent-version", default="")
    return parser


def _distribution(value) -> dict[str, float]:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            value = {
                item["label"]: item.get("probability", item.get("score", 0.0))
                for item in value
            }
        else:
            if len(value) != len(LABELS):
                raise ValueError("teacher distribution must contain six values")
            value = dict(zip(LABELS, value))
    return normalize_distribution(value)


def _load_feature(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pose feature not found: {source}")
    with np.load(source, allow_pickle=False) as artifact:
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)
        score = (np.asarray(artifact["keypoint_score"], dtype=np.float32)
                 if "keypoint_score" in artifact else None)
    valid_layout = keypoint.ndim == 4 and keypoint.shape[0] <= 2 and (
        keypoint.shape[2:] == (25, 3) or keypoint.shape[2:] == (17, 2)
    )
    if not valid_layout:
        raise ValueError(
            f"Campus6 feature must be [M,T,25,3] or [M,T,17,2], got {keypoint.shape}: {source}"
        )
    if score is not None and score.shape != keypoint.shape[:3]:
        raise ValueError(f"Campus6 keypoint_score has invalid shape: {source}")
    if keypoint.shape[1] < 1:
        raise ValueError(f"Campus6 feature has no frames: {source}")
    return keypoint, score, int(keypoint.shape[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    sample_rows: list[dict],
    *,
    pseudo_rows: list[dict] | None = None,
    replay_rows: list[dict] | None = None,
    previous_rows: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    pseudo = {
        str(row["sample_id"]): row
        for row in (pseudo_rows or [])
        if row.get("status") == "accepted"
    }
    previous = {
        str(row["sample_id"]): _distribution(
            row.get("distribution") or row.get("topk") or row.get("top5")
        )
        for row in (previous_rows or [])
    }
    merged = list(sample_rows)
    known_sample_ids = {str(row["sample_id"]) for row in merged}
    # Accepted pseudo-labels are first-class training samples.  A candidate
    # does not have to be pre-declared in the human sample manifest as long as
    # the pseudo record contains its immutable feature reference.
    for row in pseudo.values():
        sample_id = str(row["sample_id"])
        if sample_id in known_sample_ids:
            continue
        feature_path = row.get("feature_path") or row.get("artifact_path")
        if not feature_path:
            raise ValueError(
                f"accepted pseudo sample {sample_id} has no feature_path"
            )
        merged.append(
            {
                "sample_id": sample_id,
                "feature_path": feature_path,
                "source_hash": row.get("source_hash", ""),
                "label_source": "pseudo",
                "split": "train",
            }
        )
        known_sample_ids.add(sample_id)
    for row in replay_rows or []:
        value = dict(row)
        value.setdefault("label_source", "replay")
        value.setdefault("split", "train")
        sample_id = str(value["sample_id"])
        if sample_id in known_sample_ids:
            raise ValueError(f"duplicate replay sample_id: {sample_id}")
        merged.append(value)
        known_sample_ids.add(sample_id)

    seen = set()
    source_splits: dict[str, str] = {}
    annotations = []
    manifests = []
    split = {"train": [], "val": [], "test": []}
    for source_row in merged:
        row = dict(source_row)
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate dataset sample_id: {sample_id}")
        seen.add(sample_id)
        split_name = str(row.get("split", "train"))
        if split_name not in split:
            raise ValueError(f"invalid dataset split: {split_name}")
        source_hash = str(row.get("source_hash", ""))
        if source_hash:
            previous_split = source_splits.setdefault(source_hash, split_name)
            if previous_split != split_name:
                raise ValueError(
                    f"source_hash leaks across splits: {source_hash}"
                )

        pseudo_row = pseudo.get(sample_id)
        label_source = str(row.get("label_source", "human"))
        label_name = row.get("label") or row.get("manual_label")
        teacher_distribution = None
        quality_weight = 1.0
        if pseudo_row and split_name == "train":
            review = pseudo_row.get("review") or {}
            if pseudo_row.get("source") == "human_review" or review.get("final_label") in LABELS:
                label_source = "human_review"
                label_name = review.get("final_label") or pseudo_row["label"]
            elif not label_name:
                label_source = "pseudo"
                label_name = pseudo_row["label"]
            teacher_distribution = _distribution(
                pseudo_row.get("teacher_soft_label")
                or pseudo_row.get("soft_label")
            )
            quality_weight = float(pseudo_row.get("quality_score", 0.0))
        if label_name not in LABEL_TO_INDEX:
            raise ValueError(f"sample {sample_id} has no valid Campus6 label")
        if label_source == "pseudo" and split_name != "train":
            raise ValueError("automatic pseudo-labels are forbidden in val/test")

        feature_path = (
            row.get("feature_path")
            or row.get("artifact_path")
            or (pseudo_row or {}).get("feature_path")
        )
        keypoint, keypoint_score, total_frames = _load_feature(feature_path)
        hard = label_source in HARD_SOURCES
        teacher_valid = teacher_distribution is not None
        teacher_values = [
            teacher_distribution[label] if teacher_distribution else 0.0
            for label in LABELS
        ]
        previous_distribution = previous.get(sample_id)
        previous_values = [
            previous_distribution[label] if previous_distribution else 0.0
            for label in LABELS
        ]
        annotation = {
            "frame_dir": sample_id,
            "keypoint": keypoint,
            "total_frames": total_frames,
            "label": LABEL_TO_INDEX[label_name],
            "has_hard_label": int(hard),
            "teacher_distribution": np.asarray(teacher_values, dtype=np.float32),
            "teacher_valid": int(teacher_valid),
            "quality_weight": float(1.0 if hard else quality_weight),
            "previous_distribution": np.asarray(previous_values, dtype=np.float32),
            "previous_valid": int(previous_distribution is not None),
            "label_source": label_source,
            "sample_id": sample_id,
        }
        if keypoint_score is not None:
            annotation["keypoint_score"] = keypoint_score
        annotations.append(annotation)
        split[split_name].append(sample_id)
        manifests.append(
            {
                "sample_id": sample_id,
                "feature_path": str(Path(feature_path).expanduser().resolve()),
                "split": split_name,
                "label": label_name,
                "label_index": LABEL_TO_INDEX[label_name],
                "label_source": label_source,
                "has_hard_label": hard,
                "teacher_valid": teacher_valid,
                "quality_weight": annotation["quality_weight"],
                "source_hash": source_hash,
            }
        )
    return {"split": split, "annotations": annotations}, manifests


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    data, manifest = build_dataset(
        read_jsonl(args.sample_jsonl),
        pseudo_rows=read_jsonl(args.pseudo_jsonl) if args.pseudo_jsonl else None,
        replay_rows=read_jsonl(args.replay_jsonl) if args.replay_jsonl else None,
        previous_rows=read_jsonl(args.previous_jsonl) if args.previous_jsonl else None,
    )
    output = Path(args.output_dir) / args.dataset_version
    output.mkdir(parents=True, exist_ok=True)
    annotation_path = output / "annotations.pkl"
    with annotation_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in manifest
        ),
        encoding="utf-8",
    )
    counts = Counter(row["label_source"] for row in manifest)
    metadata = {
        "schema_version": "campus6_dataset.v1",
        "dataset_version": args.dataset_version,
        "parent_version": args.parent_version,
        "labels": list(LABELS),
        "samples": len(manifest),
        "splits": {name: len(values) for name, values in data["split"].items()},
        "sources": dict(sorted(counts.items())),
        "annotation_sha256": _sha256(annotation_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_event("campus6_dataset_complete", output=str(output), **metadata)


if __name__ == "__main__":
    main()
