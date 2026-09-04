"""Aggregate FP32 versus TensorRT pose extraction quality reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--predictions", required=True, type=Path,
                       help="directory containing fp32_*.json files")
    value.add_argument("--fp16-predictions", type=Path,
                       help="optional directory containing the paired fp16_*.json files")
    value.add_argument("--output", required=True, type=Path)
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    fp16_predictions = args.fp16_predictions or args.predictions
    fp32_files = sorted(
        path for path in args.predictions.glob("fp32_*.json")
        if re.fullmatch(r"fp32_[0-9]{2}\.json", path.name)
    )
    if not fp32_files:
        raise RuntimeError("no fp32_*.json files found")

    all_ious = []
    all_errors_px = []
    all_errors_norm = []
    all_score_diffs = []
    all_fp32_times = []
    all_fp16_times = []
    total_frames = total_fp32_people = total_fp16_people = 0
    matched_pairs = detections_only_fp32 = detections_only_fp16 = 0
    per_video = []

    for fp32_path in fp32_files:
        fp16_path = fp16_predictions / fp32_path.name.replace("fp32_", "fp16_", 1)
        if not fp16_path.is_file():
            raise RuntimeError("missing paired file: {}".format(fp16_path))
        fp32 = json.loads(fp32_path.read_text(encoding="utf-8"))
        fp16 = json.loads(fp16_path.read_text(encoding="utf-8"))
        fp32_frames = {frame["frame_index"]: frame for frame in fp32["frames"]}
        fp16_frames = {frame["frame_index"]: frame for frame in fp16["frames"]}
        shared = sorted(set(fp32_frames) & set(fp16_frames))
        video_ious = []
        video_errors_norm = []
        video_matched = 0
        video_only_fp32 = video_only_fp16 = 0
        for frame_index in shared:
            a_people = fp32_frames[frame_index]["people"]
            b_people = fp16_frames[frame_index]["people"]
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
                    video_only_fp32 += 1
                    continue
                paired_b.add(best[1])
                video_matched += 1
                all_ious.append(best[0])
                video_ious.append(best[0])
                a_points = np.asarray(a_person["keypoints"], dtype=np.float64)
                b_points = np.asarray(best[2]["keypoints"], dtype=np.float64)
                diagonal = float(np.linalg.norm(a_box[2:] - a_box[:2])) or 1.0
                for joint in range(17):
                    error = float(np.linalg.norm(a_points[joint] - b_points[joint]))
                    all_errors_px.append(error)
                    normalized = error / diagonal
                    all_errors_norm.append(normalized)
                    video_errors_norm.append(normalized)
                all_score_diffs.extend(
                    abs(float(a_score) - float(b_score))
                    for a_score, b_score in zip(
                        a_person.get("keypoint_scores", []),
                        best[2].get("keypoint_scores", []),
                    )
                )
            video_only_fp16 += len(b_people) - len(paired_b)

        fp32_people = sum(len(frame["people"]) for frame in fp32["frames"])
        fp16_people = sum(len(frame["people"]) for frame in fp16["frames"])
        total_frames += len(shared)
        total_fp32_people += fp32_people
        total_fp16_people += fp16_people
        matched_pairs += video_matched
        detections_only_fp32 += video_only_fp32
        detections_only_fp16 += video_only_fp16
        all_fp32_times.extend(fp32.get("inference_seconds", []))
        all_fp16_times.extend(fp16.get("inference_seconds", []))
        per_video.append({
            "id": fp32_path.stem.replace("fp32_", "", 1),
            "video": fp32.get("video"),
            "frames": len(shared),
            "fp32_people": fp32_people,
            "fp16_people": fp16_people,
            "matched_person_pairs": video_matched,
            "detections_only_fp32": video_only_fp32,
            "detections_only_fp16": video_only_fp16,
            "bbox_iou_mean": float(np.mean(video_ious)) if video_ious else None,
            "keypoint_error_normalized_median": (
                float(np.median(video_errors_norm)) if video_errors_norm else None
            ),
            "fp32_mean_inference_s": float(np.mean(fp32["inference_seconds"])),
            "fp16_mean_inference_s": float(np.mean(fp16["inference_seconds"])),
            "speedup_fp16_vs_fp32": (
                float(np.mean(fp32["inference_seconds"]) / np.mean(fp16["inference_seconds"]))
                if fp16.get("inference_seconds") and np.mean(fp16["inference_seconds"]) > 0
                else None
            ),
        })

    mean_fp32 = float(np.mean(all_fp32_times)) if all_fp32_times else None
    mean_fp16 = float(np.mean(all_fp16_times)) if all_fp16_times else None
    report = {
        "videos": len(per_video),
        "frames_compared": total_frames,
        "fp32_people": total_fp32_people,
        "fp16_people": total_fp16_people,
        "matched_person_pairs": matched_pairs,
        "detections_only_fp32": detections_only_fp32,
        "detections_only_fp16": detections_only_fp16,
        "detection_recall_vs_fp32": (
            float(matched_pairs / total_fp32_people) if total_fp32_people else None
        ),
        "detection_precision_vs_fp32": (
            float(matched_pairs / total_fp16_people) if total_fp16_people else None
        ),
        "bbox_iou_mean": float(np.mean(all_ious)) if all_ious else None,
        "bbox_iou_median": float(np.median(all_ious)) if all_ious else None,
        "keypoint_error_px_mean": float(np.mean(all_errors_px)) if all_errors_px else None,
        "keypoint_error_px_median": float(np.median(all_errors_px)) if all_errors_px else None,
        "keypoint_error_normalized_mean": (
            float(np.mean(all_errors_norm)) if all_errors_norm else None
        ),
        "keypoint_error_normalized_median": (
            float(np.median(all_errors_norm)) if all_errors_norm else None
        ),
        "keypoint_score_diff_mean_abs": (
            float(np.mean(all_score_diffs)) if all_score_diffs else None
        ),
        "fp32_mean_inference_s": mean_fp32,
        "fp16_mean_inference_s": mean_fp16,
        "speedup_fp16_vs_fp32": (
            float(mean_fp32 / mean_fp16) if mean_fp32 and mean_fp16 else None
        ),
        "per_video": per_video,
    }
    report["quality_gate"] = {
        "accepted": bool(
            matched_pairs > 0
            and (report["bbox_iou_mean"] or 0.0) >= 0.8
            and (report["keypoint_error_normalized_median"] or float("inf")) <= 2.5e-2
        ),
        "criteria": {
            "bbox_iou_mean_min": 0.8,
            "keypoint_error_normalized_median_max": 2.5e-2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
