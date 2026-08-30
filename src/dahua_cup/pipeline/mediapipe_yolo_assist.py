"""YOLO-person assisted MediaPipe Pose extraction for difficult two-person clips."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from dahua_cup.feature_extraction.mediapipe_pose import mediapipe33_to_ntu25
from dahua_cup.pipeline.common import log_event, require_file


def _landmarks(points):
    return np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)


def _visibility(points):
    return np.asarray([float(getattr(p, "visibility", 1.0)) for p in points], dtype=np.float32)


def _make_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.GPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def extract(video_path: Path, model_path: Path, yolo_path: Path):
    from ultralytics import YOLO

    mp, landmarker = _make_landmarker(model_path)
    detector = YOLO(str(yolo_path))
    capture = cv2.VideoCapture(str(require_file(video_path, "input video")))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames, scores, masks = [], [], []
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            prediction = detector.predict(
                bgr, classes=[0], conf=0.15, iou=0.5, max_det=2,
                device=0, verbose=False,
            )[0]
            boxes = []
            if prediction.boxes is not None:
                for box, conf in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    margin = int(0.08 * max(x2 - x1, y2 - y1))
                    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                    x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
                    if x2 - x1 >= 32 and y2 - y1 >= 64:
                        boxes.append((x1, y1, x2, y2, float(conf)))
            boxes.sort(key=lambda item: item[0])
            keypoint = np.zeros((2, 25, 3), dtype=np.float32)
            score = np.zeros((2, 25), dtype=np.float32)
            valid = np.zeros((2, 25), dtype=bool)
            for slot, (x1, y1, x2, y2, _) in enumerate(boxes[:2]):
                crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                timestamp = frame_index * 2 + slot
                result = landmarker.detect_for_video(image, timestamp)
                if not result.pose_world_landmarks:
                    continue
                world = _landmarks(result.pose_world_landmarks[0])
                visibility = _visibility(result.pose_landmarks[0]) if result.pose_landmarks else np.ones(33, dtype=np.float32)
                world[:, 1] *= -1.0
                mapped, mapped_score, mapped_valid = mediapipe33_to_ntu25(
                    world, visibility, 0.1
                )
                keypoint[slot] = mapped
                score[slot] = mapped_score
                valid[slot] = mapped_valid
            frames.append(keypoint)
            scores.append(score)
            masks.append(valid)
            frame_index += 1
    finally:
        capture.release()
        landmarker.close()
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {video_path}")
    return {
        "schema_version": np.asarray("mediapipe_yolo_assist_ntu25.v1"),
        "keypoint": np.stack(frames, axis=1),
        "keypoint_score": np.stack(scores, axis=1),
        "valid_mask": np.stack(masks, axis=1),
        "fps": np.asarray(fps, dtype=np.float32),
        "total_frames": np.asarray(frame_index, dtype=np.int32),
        "delegate": np.asarray("gpu"),
        "physical_gpu_id": np.asarray(os.environ.get("DAHUA_PHYSICAL_GPU_ID", "")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--feature", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--yolo", required=True, type=Path)
    args = parser.parse_args()
    arrays = extract(args.video, args.model, args.yolo)
    args.feature.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.feature, **arrays)
    log_event(
        "mediapipe_yolo_assist_complete", video=str(args.video), feature=str(args.feature),
        frames=int(arrays["total_frames"]), valid_joints=int(np.count_nonzero(arrays["valid_mask"])),
    )


if __name__ == "__main__":
    main()
