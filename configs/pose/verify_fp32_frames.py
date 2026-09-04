"""Extract FP32 RTMDet-S + RTMPose-S per-frame predictions for INT8 comparison.

Runs the shipped S weights through the production MMPose inferencer on a fixed
frame selection and writes per-frame person bboxes and COCO-17 keypoints (raw
pixels) plus wall-clock inference timing to a JSON file.

Usage (in the FP32 pose environment):

    <fp32-env>/bin/python configs/pose/verify_fp32_frames.py \
      --video /path/to/video.avi \
      --pose2d-weights models/pose/rtmpose-s_coco17.pth \
      --det-weights models/pose/rtmdet-s_coco80.pth \
      --output /path/to/fp32_predictions.json \
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
    value.add_argument("--pose2d-weights", required=True)
    value.add_argument("--det-weights", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--max-frames", type=int, default=30)
    value.add_argument("--score", type=float, default=0.15)
    value.add_argument(
        "--det-config",
        help="optional detector config; otherwise use the packaged RTMDet-S config",
    )
    return value


def installed_config(component: str, checkpoint: Path) -> Path:
    """Return the ABI-compatible packaged config for the shipped S weights."""
    name = checkpoint.name.lower()
    if component == "pose" and name == "rtmpose-s_coco17.pth":
        import mmpose
        base = Path(mmpose.__file__).resolve().parent
        return base / ".mim" / "configs" / "body_2d_keypoint" / "rtmpose" / "coco" / \
            "rtmpose-s_8xb256-420e_coco-256x192.py"
    if component == "detector" and name == "rtmdet-s_coco80.pth":
        import mmdet
        base = Path(mmdet.__file__).resolve().parent
        return base / ".mim" / "configs" / "rtmdet" / "rtmdet_s_8xb32-300e_coco.py"
    raise ValueError(f"no packaged config for {checkpoint}")


def frame_indices(video: Path, max_frames: int) -> list[int]:
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if total <= max_frames:
        return list(range(total))
    return np.linspace(0, total - 1, max_frames, dtype=np.int32).tolist()


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    from mmpose.apis.inferencers import Pose2DInferencer

    pose_weights = Path(args.pose2d_weights)
    det_weights = Path(args.det_weights)
    detector_config = args.det_config or str(installed_config("detector", det_weights))
    inferencer = Pose2DInferencer(
        model=str(installed_config("pose", pose_weights)),
        weights=str(pose_weights),
        det_model=detector_config,
        det_weights=str(det_weights),
        det_cat_ids=[0],
        device=args.device,
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
        result = next(inferencer(image, return_vis=False, bbox_thr=args.score, nms_thr=0.5))
        timings.append(time.perf_counter() - started)
        height, width = image.shape[:2]
        people = []
        for person in (result.get("predictions") or [[]])[0]:
            bbox = np.asarray(person.get("bbox", [[0, 0, 0, 0]]), dtype=np.float32).reshape(-1, 4)[0]
            score = float(np.asarray(person.get("bbox_score", [1.0])).reshape(-1)[0])
            if score < args.score:
                continue
            points = np.asarray(person.get("keypoints", []), dtype=np.float32).reshape(-1, 2)
            scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32).reshape(-1)
            if points.shape != (17, 2):
                continue
            people.append({
                "bbox": bbox.tolist(),
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
        "backend": "mmpose_fp32",
        "video": str(Path(args.video)),
        "bbox_score": args.score,
        "frames": frames,
        "inference_seconds": timings,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"frames={len(frames)} people={sum(len(f['people']) for f in frames)} "
          f"mean_inference_s={float(np.mean(timings)):.4f}")


if __name__ == "__main__":
    main()
