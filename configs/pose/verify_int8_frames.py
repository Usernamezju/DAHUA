"""Extract TensorRT RTMDet + RTMPose per-frame predictions for comparison.

Runs the exported TensorRT INT8 engines through the production MMDeploy
runtime on the same frame selection as ``verify_fp32_frames.py`` and writes
the same JSON schema plus wall-clock inference timing.

Usage (in the INT8 export environment):

    LD_LIBRARY_PATH=<tensorrt_libs> \
    <int8-env>/bin/python configs/pose/verify_int8_frames.py \
      --video /path/to/video.avi \
      --detector-model-dir models/pose/int8/rtmdet_s \
      --pose-model-dir models/pose/int8/rtmpose_s \
      --output /path/to/int8_predictions.json \
      --device cuda:0 --max-frames 30 --score 0.15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--video", required=True)
    value.add_argument("--detector-model-dir", required=True)
    value.add_argument("--pose-model-dir", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--max-frames", type=int, default=30)
    value.add_argument("--score", type=float, default=0.15)
    value.add_argument("--backend-label", default="tensorrt_int8")
    return value


def frame_indices(video: Path, max_frames: int) -> list[int]:
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if total <= max_frames:
        return list(range(total))
    return np.linspace(0, total - 1, max_frames, dtype=np.int32).tolist()


def nms_keep(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Replicate MMPose's score-filtered greedy NMS for MMDeploy outputs."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    coordinates = boxes[:, :4].astype(np.float64, copy=False)
    scores = boxes[:, 4].astype(np.float64, copy=False)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(coordinates[current, 0], coordinates[rest, 0])
        y1 = np.maximum(coordinates[current, 1], coordinates[rest, 1])
        x2 = np.minimum(coordinates[current, 2], coordinates[rest, 2])
        y2 = np.minimum(coordinates[current, 3], coordinates[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = np.maximum(0.0, coordinates[current, 2] - coordinates[current, 0]) * np.maximum(
            0.0, coordinates[current, 3] - coordinates[current, 1]
        )
        area_rest = np.maximum(0.0, coordinates[rest, 2] - coordinates[rest, 0]) * np.maximum(
            0.0, coordinates[rest, 3] - coordinates[rest, 1]
        )
        union = area_current + area_rest - intersection
        overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    from mmdeploy_runtime import Detector, PoseDetector

    kind, _, device_id = args.device.partition(":")
    detector = Detector(
        args.detector_model_dir,
        device_name=kind or "cuda",
        device_id=int(device_id or 0),
    )
    pose_detector = PoseDetector(
        args.pose_model_dir,
        device_name=kind or "cuda",
        device_id=int(device_id or 0),
    )

    capture = cv2.VideoCapture(str(Path(args.video)))
    wanted = set(frame_indices(Path(args.video), args.max_frames))
    frames = []
    source_frame = 0
    timings = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if source_frame not in wanted:
            source_frame += 1
            continue
        started = time.perf_counter()
        detection = detector(image)
        bboxes = np.asarray(detection[0], dtype=np.float32)
        labels = np.asarray(detection[1]).reshape(-1)
        person_boxes = bboxes[labels == 0] if bboxes.size else np.zeros((0, 5), dtype=np.float32)
        if person_boxes.shape[0]:
            person_boxes = person_boxes[person_boxes[:, 4] > args.score]
            person_boxes = person_boxes[nms_keep(person_boxes, 0.5)]
        keypoints = (
            pose_detector(image, person_boxes[:, :4]) if person_boxes.shape[0] else np.zeros((0, 17, 3), dtype=np.float32)
        )
        timings.append(time.perf_counter() - started)
        values = np.asarray(keypoints, dtype=np.float32)
        if values.ndim == 2:
            values = values[None, ...]
        height, width = image.shape[:2]
        people = []
        for index, box in enumerate(person_boxes):
            score = float(box[4])
            if score < args.score or index >= values.shape[0]:
                continue
            points = values[index, :, :2]
            scores = values[index, :, 2] if values.shape[-1] >= 3 else np.ones(17, dtype=np.float32)
            if points.shape != (17, 2):
                continue
            people.append({
                "bbox": [float(v) for v in box[:4]],
                "bbox_score": score,
                "keypoints": points.tolist(),
                "keypoint_scores": scores.tolist(),
            })
        frames.append({
            "frame_index": source_frame,
            "width": width,
            "height": height,
            "people": people,
        })
        source_frame += 1
    capture.release()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "backend": args.backend_label,
        "video": str(Path(args.video)),
        "bbox_score": args.score,
        "frames": frames,
        "inference_seconds": timings,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"frames={len(frames)} people={sum(len(f['people']) for f in frames)} "
          f"mean_inference_s={float(np.mean(timings)):.4f}")


if __name__ == "__main__":
    main()
