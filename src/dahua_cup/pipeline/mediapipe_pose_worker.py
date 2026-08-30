"""Extract MediaPipe 3D poses from one video as ProtoGCN NTU-25 input."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from dahua_cup.feature_extraction.mediapipe_pose import (
    assign_pose_slots,
    mediapipe33_to_ntu25,
    normalized_pose_center,
)
from dahua_cup.pipeline.common import log_event, require_file


SCHEMA_VERSION = "mediapipe_ntu25.v1"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("DAHUA_MEDIAPIPE_MODEL"),
        help="Pose Landmarker .task file (or DAHUA_MEDIAPIPE_MODEL)",
    )
    parser.add_argument("--num-poses", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--delegate",
        choices=("cpu", "gpu"),
        default=os.environ.get("DAHUA_MEDIAPIPE_DELEGATE", "cpu").lower(),
        help="MediaPipe inference delegate; the Web pipeline defaults to CPU",
    )
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--joint-score-threshold", type=float, default=0.2)
    return parser


def _landmark_array(landmarks):
    return np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)


def _visibility_array(landmarks):
    return np.asarray(
        [float(getattr(point, "visibility", 1.0)) for point in landmarks],
        dtype=np.float32,
    )


def create_landmarker(args):
    """Create one reusable landmarker for a sequence of compatible videos."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe and OpenCV are required; install mediapipe and opencv-python"
        ) from exc

    if not args.model:
        raise ValueError("--model or DAHUA_MEDIAPIPE_MODEL is required")
    model_path = require_file(args.model, "MediaPipe Pose Landmarker model")
    delegate = getattr(mp_python.BaseOptions.Delegate, args.delegate.upper())
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def extract_video_with_landmarker(args, mp, landmarker):
    """Extract one video using an already initialized MediaPipe landmarker."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install opencv-python") from exc

    video_path = require_file(args.video, "input video")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"video reports invalid FPS: {fps}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_keypoints = []
    frame_image_keypoints = []
    frame_scores = []
    frame_validity = []
    previous_centers = [None] * args.num_poses
    frame_index = 0
    last_timestamp_ms = -1

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Some container FPS values are rounded or malformed.  The Tasks
            # VIDEO API requires strict monotonicity, so retain the nominal
            # frame clock but guarantee at least a 1 ms advance per decoded
            # frame.
            timestamp_ms = max(
                last_timestamp_ms + 1,
                int(round(frame_index * 1000.0 / fps)),
            )
            result = landmarker.detect_for_video(image, timestamp_ms)
            last_timestamp_ms = timestamp_ms

            normalized_poses = list(result.pose_landmarks or [])[: args.num_poses]
            world_poses = list(result.pose_world_landmarks or [])[: args.num_poses]
            detected_count = min(len(normalized_poses), len(world_poses))
            normalized_poses = normalized_poses[:detected_count]
            world_poses = world_poses[:detected_count]

            centers = [
                normalized_pose_center(_landmark_array(pose))
                for pose in normalized_poses
            ]
            slots = assign_pose_slots(centers, previous_centers, args.num_poses)
            keypoints = np.zeros((args.num_poses, 25, 3), dtype=np.float32)
            image_keypoints = np.zeros((args.num_poses, 25, 2), dtype=np.float32)
            scores = np.zeros((args.num_poses, 25), dtype=np.float32)
            validity = np.zeros((args.num_poses, 25), dtype=bool)

            next_centers = list(previous_centers)
            for detection_index, slot in enumerate(slots):
                normalized = _landmark_array(normalized_poses[detection_index])
                world = _landmark_array(world_poses[detection_index])
                # MediaPipe's vertical axis points down; NTU camera-space Y
                # points up. Keep X and relative depth, and invert Y.
                world[:, 1] *= -1.0
                visibility = _visibility_array(normalized_poses[detection_index])
                mapped, mapped_scores, valid = mediapipe33_to_ntu25(
                    world, visibility, args.joint_score_threshold
                )
                mapped_image, _, _ = mediapipe33_to_ntu25(
                    normalized, visibility, args.joint_score_threshold
                )
                keypoints[slot] = mapped
                image_keypoints[slot, :, 0] = mapped_image[:, 0] * width
                image_keypoints[slot, :, 1] = mapped_image[:, 1] * height
                scores[slot] = mapped_scores
                validity[slot] = valid
                next_centers[slot] = centers[detection_index]

            previous_centers = next_centers
            frame_keypoints.append(keypoints)
            frame_image_keypoints.append(image_keypoints)
            frame_scores.append(scores)
            frame_validity.append(validity)
            frame_index += 1
    finally:
        capture.release()

    if not frame_keypoints:
        raise RuntimeError(f"video has no decodable frames: {video_path}")

    # Frame-major lists become the M,T,V,C convention expected by ProtoGCN.
    keypoint = np.stack(frame_keypoints, axis=1)
    keypoint_score = np.stack(frame_scores, axis=1)
    valid_mask = np.stack(frame_validity, axis=1)
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "keypoint": keypoint,
        "image_keypoint": np.stack(frame_image_keypoints, axis=1),
        "keypoint_score": keypoint_score,
        "valid_mask": valid_mask,
        "fps": np.asarray(fps, dtype=np.float32),
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
        "total_frames": np.asarray(frame_index, dtype=np.int32),
        "delegate": np.asarray(args.delegate),
        "physical_gpu_id": np.asarray(
            os.environ.get("DAHUA_PHYSICAL_GPU_ID", "")
        ),
    }


def extract_video(args):
    """Extract one video, retaining the original single-video CLI contract."""
    mp, landmarker = create_landmarker(args)
    try:
        return extract_video_with_landmarker(args, mp, landmarker)
    finally:
        landmarker.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    thresholds = (
        args.min_pose_detection_confidence,
        args.min_pose_presence_confidence,
        args.min_tracking_confidence,
        args.joint_score_threshold,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("MediaPipe confidence thresholds must be in [0, 1]")
    arrays = extract_video(args)
    destination = Path(args.feature)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    detected = int(np.count_nonzero(arrays["valid_mask"]))
    log_event(
        "mediapipe_pose_complete",
        video=str(args.video),
        feature=str(destination),
        frames=int(arrays["total_frames"]),
        valid_joints=detected,
        delegate=str(arrays["delegate"]),
        physical_gpu_id=str(arrays["physical_gpu_id"]),
    )


if __name__ == "__main__":
    main()
