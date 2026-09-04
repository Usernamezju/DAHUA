"""Compare FP32 vs INT8 RTM pose predictions produced by the verify scripts.

Matches persons per frame by bbox IoU (greedy, threshold 0.5) and reports:
detection agreement, bbox IoU, per-joint keypoint L2 error in pixels and
normalized by bbox diagonal, joint-score agreement, and the wall-clock
inference speedup. Exits 0 when the INT8 output is within the acceptance
budget (median per-joint normalized error <= 2.5e-2, mean IoU >= 0.8).

Usage:

    python configs/pose/compare_pose_outputs.py \
      --fp32 /path/to/fp32_predictions.json \
      --int8 /path/to/int8_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fp32", required=True)
    value.add_argument("--int8", required=True)
    return value


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    fp32 = json.loads(Path(args.fp32).read_text(encoding="utf-8"))
    int8 = json.loads(Path(args.int8).read_text(encoding="utf-8"))
    fp32_frames = {f["frame_index"]: f for f in fp32["frames"]}
    int8_frames = {f["frame_index"]: f for f in int8["frames"]}
    shared = sorted(set(fp32_frames) & set(int8_frames))
    if not shared:
        raise RuntimeError("no shared frames between the two prediction files")

    ious, errors_px, errors_norm, score_diffs = [], [], [], []
    matched_pairs = det_only_fp32 = det_only_int8 = 0
    per_joint_errors = [[] for _ in range(17)]
    for frame_index in shared:
        a_people = fp32_frames[frame_index]["people"]
        b_people = int8_frames[frame_index]["people"]
        paired_b = set()
        for a_person in a_people:
            a_box = np.asarray(a_person["bbox"], dtype=np.float64)
            best = None
            for b_index, b_person in enumerate(b_people):
                if b_index in paired_b:
                    continue
                iou = box_iou(a_box, np.asarray(b_person["bbox"], dtype=np.float64))
                if iou >= 0.5 and (best is None or iou > best[0]):
                    best = (iou, b_index, b_person)
            if best is None:
                det_only_fp32 += 1
                continue
            paired_b.add(best[1])
            ious.append(best[0])
            a_points = np.asarray(a_person["keypoints"], dtype=np.float64)
            b_points = np.asarray(best[2]["keypoints"], dtype=np.float64)
            diagonal = float(np.linalg.norm(a_box[2:] - a_box[:2])) or 1.0
            for joint in range(17):
                distance = float(np.linalg.norm(a_points[joint] - b_points[joint]))
                per_joint_errors[joint].append(distance)
                errors_px.append(distance)
                errors_norm.append(distance / diagonal)
            score_diffs.extend(
                float(s) for s in (
                    np.asarray(a_person["keypoint_scores"], dtype=np.float64)
                    - np.asarray(best[2]["keypoint_scores"], dtype=np.float64)
                )
            )
            matched_pairs += 1
        det_only_int8 += len(b_people) - len(paired_b)

    ious = np.asarray(ious)
    errors_px = np.asarray(errors_px)
    errors_norm = np.asarray(errors_norm)
    per_joint_median = np.array([
        float(np.median(per_joint_errors[joint])) if per_joint_errors[joint] else float("nan")
        for joint in range(17)
    ])
    speedup = (
        float(np.mean(fp32["inference_seconds"]) / np.mean(int8["inference_seconds"]))
        if int8.get("inference_seconds") else float("nan")
    )

    report = {
        "frames_compared": len(shared),
        "matched_person_pairs": matched_pairs,
        "detections_only_fp32": det_only_fp32,
        "detections_only_int8": det_only_int8,
        "bbox_iou": {
            "mean": float(ious.mean()) if ious.size else float("nan"),
            "min": float(ious.min()) if ious.size else float("nan"),
        },
        "keypoint_error_px": {
            "mean": float(errors_px.mean()) if errors_px.size else float("nan"),
            "median": float(np.median(errors_px)) if errors_px.size else float("nan"),
        },
        "keypoint_error_normalized": {
            "mean": float(errors_norm.mean()) if errors_norm.size else float("nan"),
            "median": float(np.median(errors_norm)) if errors_norm.size else float("nan"),
        },
        "keypoint_error_median_by_joint_px": per_joint_median.tolist(),
        "keypoint_score_diff_mean": float(np.mean(np.abs(score_diffs))) if score_diffs else float("nan"),
        "inference_speedup_int8_vs_fp32": speedup,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    acceptance = (
        matched_pairs > 0
        and (not np.isnan(report["bbox_iou"]["mean"]) and report["bbox_iou"]["mean"] >= 0.8)
        and (not np.isnan(report["keypoint_error_normalized"]["median"]) and report["keypoint_error_normalized"]["median"] <= 2.5e-2)
    )
    print("ACCEPTED" if acceptance else "REJECTED", file=sys.stderr)
    return 0 if acceptance else 1


if __name__ == "__main__":
    sys.exit(main())
