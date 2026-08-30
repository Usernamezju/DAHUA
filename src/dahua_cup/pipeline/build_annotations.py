"""Build grouped Campus6 annotations from RTMPose Halpe-26 outputs.

The first 17 Halpe joints use the COCO ordering expected by ProtoGCN's
``Kinetics_Transform``.  That transform constructs the remaining three
``coco_new`` torso joints inside the training pipeline, so this file keeps the
raw COCO-17 coordinates and their confidence scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
INTERACTION_IDS = set(range(2, len(LABELS)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--test-ratio", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--min-primary-coverage", type=float, default=.10)
    parser.add_argument("--min-secondary-coverage", type=float, default=.03)
    return parser.parse_args()


def feature_path(root: Path, row: dict) -> Path:
    video = Path(row["video"])
    # Must match extract_rtmpose26_batch.py exactly (it uses str(video), not
    # Path.resolve(), so the result remains stable across mounted filesystems).
    digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
    return root / row["label"] / f"{video.stem}__{digest}.npz"


def group_for(row: dict) -> str:
    value = row.get("parent_video_id") or row.get("source_group")
    if value:
        return str(value)
    return str(Path(row["video"]).with_suffix(""))


def select_split(grouped, val_ratio, test_ratio, rng):
    train_groups, val_groups, test_groups, summary = set(), set(), set(), {}
    for label, source_rows in sorted(grouped.items()):
        groups = sorted(source_rows)
        rng.shuffle(groups)
        count = len(groups)
        test_count = min(max(1, round(count * test_ratio)), max(0, count - 1))
        val_count = min(max(1, round(count * val_ratio)), max(0, count - test_count - 1))
        selected_test = set(groups[:test_count])
        selected_val = set(groups[test_count:test_count + val_count])
        selected_train = set(groups) - selected_test - selected_val
        train_groups.update((label, group) for group in selected_train)
        val_groups.update((label, group) for group in selected_val)
        test_groups.update((label, group) for group in selected_test)
        summary[label] = {
            "groups": count, "train_groups": len(selected_train),
            "val_groups": len(selected_val), "test_groups": len(selected_test),
            "train_samples": sum(len(source_rows[x]) for x in selected_train),
            "val_samples": sum(len(source_rows[x]) for x in selected_val),
            "test_samples": sum(len(source_rows[x]) for x in selected_test),
        }
    return train_groups, val_groups, test_groups, summary


def main():
    args = parse_args()
    if not 0 <= args.val_ratio < .5 or not 0 < args.test_ratio < .5:
        raise ValueError("validation/test ratios must be in (valid) [0, .5) ranges")
    rows, discarded, missing, coverage = [], [], [], defaultdict(list)
    for raw_line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        label = item["label"]
        if label not in LABEL_TO_ID:
            continue
        path = feature_path(args.features, item)
        if not path.is_file():
            missing.append(str(path))
            continue
        with np.load(path) as values:
            keypoint = np.asarray(values["keypoint"], dtype=np.float32)
            score = np.asarray(values["keypoint_score"], dtype=np.float32)
            total_frames = int(values["total_frames"])
        if keypoint.ndim != 4 or keypoint.shape[0] != 2 or keypoint.shape[2:] != (26, 2):
            raise ValueError(f"unexpected RTMPose tensor {path}: {keypoint.shape}")
        if score.shape != keypoint.shape[:3]:
            raise ValueError(f"unexpected score tensor {path}: {score.shape}")
        keypoint, score = keypoint[:, :, :17], score[:, :, :17]
        person_coverage = (score > .10).mean(axis=(1, 2))
        primary, secondary = map(float, person_coverage)
        label_id = LABEL_TO_ID[label]
        coverage[label].append({"primary": primary, "secondary": secondary})
        if primary < args.min_primary_coverage or (
                label_id in INTERACTION_IDS and secondary < args.min_secondary_coverage):
            discarded.append({"video": item["video"], "label": label,
                              "primary": primary, "secondary": secondary})
            continue
        sample_id = f"{label}/{path.stem}"
        rows.append({
            "frame_dir": sample_id, "total_frames": total_frames,
            "label": label_id, "keypoint": keypoint, "keypoint_score": score,
            # RTMPose coordinates are already divided by frame width/height.
            # The Kinetics horizontal-flip transform therefore needs the
            # unit image canvas so x becomes 1 - x rather than failing on a
            # missing source-resolution field.
            "img_shape": (1, 1),
            "source_group": group_for(item),
        })
    if missing:
        raise RuntimeError(f"{len(missing)} features missing; example: {missing[0]}")
    if not rows:
        raise RuntimeError("no usable RTMPose feature rows")

    by_label_group = defaultdict(lambda: defaultdict(list))
    for item in rows:
        by_label_group[LABELS[item["label"]]][item["source_group"]].append(item)
    train_groups, val_groups, test_groups, split_summary = select_split(
        by_label_group, args.val_ratio, args.test_ratio, random.Random(args.seed))
    annotations, split = [], {"train": [], "val": [], "test": []}
    for item in rows:
        group = item.pop("source_group")
        label = LABELS[item["label"]]
        annotations.append(item)
        key = (label, group)
        split["test" if key in test_groups else "val" if key in val_groups else "train"].append(item["frame_dir"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump({"split": split, "annotations": annotations}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "labels": list(LABELS), "samples": len(annotations),
        "train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"]),
        "class_samples": dict(sorted(Counter(LABELS[x["label"]] for x in annotations).items())),
        "discarded_by_pose_gate": dict(sorted(Counter(x["label"] for x in discarded).items())),
        "split_by_class": split_summary, "seed": args.seed,
        "min_primary_coverage": args.min_primary_coverage,
        "min_secondary_coverage": args.min_secondary_coverage,
        "coverage_mean": {key: {"primary": float(np.mean([x["primary"] for x in values])),
                                 "secondary": float(np.mean([x["secondary"] for x in values]))}
                          for key, values in coverage.items()},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".discarded.json").write_text(json.dumps(discarded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
