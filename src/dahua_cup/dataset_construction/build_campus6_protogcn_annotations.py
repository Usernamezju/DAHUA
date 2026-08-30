#!/usr/bin/env python3
"""Build a leakage-safe ProtoGCN annotation pickle from Campus6 NTU25 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LABELS = (
    "normal_walk",
    "normal_run",
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def feature_path(features: Path, video: Path, label: str) -> Path:
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:12]
    return features / label / (video.stem + "__" + digest + ".npz")


def source_group(video: Path) -> str:
    """Identify a long-video source without retaining its 5/10-second clip id."""
    stem = re.sub(r"_clip\d+$", "", video.stem)
    return str(video.parent / stem)


def read_manifest(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main():
    args = parse_args()
    if not 0 <= args.val_ratio < 0.5 or not 0 < args.test_ratio < 0.5:
        raise ValueError("--val-ratio must be in [0, 0.5) and --test-ratio in (0, 0.5)")
    rows = []
    missing = []
    for row in read_manifest(args.manifest):
        label = row["label"]
        if label not in LABEL_TO_ID:
            raise ValueError("unknown label: %s" % label)
        video = Path(row["video"])
        feature = feature_path(args.features, video, label)
        if not feature.is_file():
            missing.append(str(feature))
            continue
        with np.load(feature) as arrays:
            keypoint = np.asarray(arrays["keypoint"], dtype=np.float32)
            total_frames = int(arrays["total_frames"])
        if keypoint.ndim != 4 or keypoint.shape[0] not in (1, 2) or keypoint.shape[2:] != (25, 3):
            raise ValueError("unexpected NTU25 tensor: %s %s" % (feature, keypoint.shape))
        sample_id = "%s/%s" % (label, feature.stem)
        rows.append({
            "frame_dir": sample_id,
            "total_frames": total_frames,
            "label": LABEL_TO_ID[label],
            "keypoint": keypoint,
            "source_group": source_group(video),
        })
    if missing:
        raise RuntimeError("%d feature files are missing; example: %s" % (len(missing), missing[0]))
    if not rows:
        raise RuntimeError("no feature rows found")

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["label"]][row["source_group"]].append(row)
    rng = random.Random(args.seed)
    val_groups, test_groups = set(), set()
    split_counts = {}
    for label_id, source_rows in grouped.items():
        groups = sorted(source_rows)
        rng.shuffle(groups)
        # Split at the long-video level.  Keep at least one source group for
        # training and, whenever possible, a genuinely disjoint test group.
        n_groups = len(groups)
        test_count = min(max(1, int(round(n_groups * args.test_ratio))), max(0, n_groups - 1))
        val_count = min(max(1, int(round(n_groups * args.val_ratio))) if args.val_ratio else 0,
                        max(0, n_groups - test_count - 1))
        test_chosen = set(groups[:test_count])
        val_chosen = set(groups[test_count:test_count + val_count])
        test_groups.update(test_chosen)
        val_groups.update(val_chosen)
        split_counts[LABELS[label_id]] = {
            "groups": n_groups,
            "train_groups": n_groups - len(test_chosen) - len(val_chosen),
            "val_groups": len(val_chosen),
            "test_groups": len(test_chosen),
            "train_samples": sum(len(source_rows[g]) for g in groups if g not in test_chosen | val_chosen),
            "val_samples": sum(len(source_rows[g]) for g in val_chosen),
            "test_samples": sum(len(source_rows[g]) for g in test_chosen),
        }

    annotations = []
    train_ids, val_ids, test_ids = [], [], []
    for row in rows:
        source = row.pop("source_group")
        annotations.append(row)
        if source in test_groups:
            test_ids.append(row["frame_dir"])
        elif source in val_groups:
            val_ids.append(row["frame_dir"])
        else:
            train_ids.append(row["frame_dir"])
    payload = {
        "split": {"train": train_ids, "val": val_ids, "test": test_ids},
        "annotations": annotations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "labels": list(LABELS), "samples": len(annotations),
        "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
        "class_samples": dict(sorted(Counter(LABELS[row["label"]] for row in annotations).items())),
        "split_by_class": split_counts,
        "seed": args.seed,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
