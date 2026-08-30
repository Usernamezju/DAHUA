"""Create an auditable per-clip skeleton-quality manifest for Campus6."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


INTERACTION_LABELS = {
    "playful_chase", "playful_push", "conflict_chase", "conflict_push"
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def feature_path(features: Path, video: Path, label: str) -> Path:
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:12]
    return features / label / (video.stem + "__" + digest + ".npz")


def main():
    args = parse_args()
    output_rows = []
    summary = defaultdict(Counter)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        video = Path(source["video"])
        label = source["label"]
        feature = feature_path(args.features, video, label)
        row = {"label": label, "video": str(video), "feature": str(feature)}
        reasons = []
        if not feature.is_file():
            reasons.append("missing_feature")
            row.update({"valid_frame_ratio": 0.0, "two_person_frame_ratio": 0.0})
        else:
            with np.load(feature) as arrays:
                keypoint = np.asarray(arrays["keypoint"], dtype=np.float32)
                total_frames = int(arrays["total_frames"]) if "total_frames" in arrays else keypoint.shape[1]
            present = np.abs(keypoint).sum(axis=(2, 3)) > 1e-7  # M,T
            valid = present.any(axis=0)
            two_people = present.all(axis=0)
            row.update({
                "total_frames": total_frames,
                "valid_frame_ratio": float(valid.mean()),
                "two_person_frame_ratio": float(two_people.mean()),
                "person0_visible_frames": int(present[0].sum()),
                "person1_visible_frames": int(present[1].sum()) if keypoint.shape[0] > 1 else 0,
            })
            if not valid.any():
                reasons.append("empty_skeleton")
            elif valid.mean() < 0.7:
                reasons.append("low_pose_coverage")
            if label in INTERACTION_LABELS and two_people.mean() < 0.5:
                reasons.append("interaction_has_under_50pct_two_person_coverage")
        row["review_reasons"] = reasons
        row["review_priority"] = (
            "high" if label == "conflict_chase" or "empty_skeleton" in reasons
            else "medium" if reasons else "low"
        )
        output_rows.append(row)
        summary[label]["clips"] += 1
        summary[label]["review"] += bool(reasons)
        summary[label]["high_priority"] += row["review_priority"] == "high"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
