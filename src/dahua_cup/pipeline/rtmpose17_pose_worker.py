"""Extract a single video as the Campus6 direct RTMPose COCO-17 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.common import log_event, require_file
from dahua_cup.pipeline.extract_rtmpose26_batch import GreedyTracker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--video", required=True)
    value.add_argument("--feature", required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--frame-stride", type=int, default=1)
    value.add_argument("--max-frames", type=int, default=100)
    value.add_argument("--bbox-score", type=float, default=0.15)
    value.add_argument("--joint-score-threshold", type=float, default=0.20)
    value.add_argument("--pose2d", default="human")
    return value


def _detections(result, width: int, height: int, threshold: float):
    people = (result.get("predictions") or [[]])[0]
    parsed = []
    for person in people:
        bbox = np.asarray(person.get("bbox", [[0, 0, 0, 0]]), dtype=np.float32).reshape(-1, 4)[0]
        detection_score = float(np.asarray(person.get("bbox_score", [1.0])).reshape(-1)[0])
        points = np.asarray(person.get("keypoints", []), dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32).reshape(-1)
        if points.shape != (17, 2) or scores.shape != (17,) or detection_score < threshold:
            continue
        points[:, 0] /= max(width, 1)
        points[:, 1] /= max(height, 1)
        parsed.append((bbox, points, scores, detection_score))
    return parsed


def extract(video: Path, inferencer, *, frame_stride: int, max_frames: int, bbox_score: float,
            joint_score_threshold: float) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = set(np.linspace(0, source_total - 1, max_frames, dtype=np.int32).tolist()) if max_frames > 0 and source_total > max_frames else None
    tracker, frame, source_frame, width, height = GreedyTracker(max_gap=6 * frame_stride), 0, 0, 0, 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_frame % frame_stride or (selected is not None and source_frame not in selected):
                source_frame += 1
                continue
            height, width = image.shape[:2]
            result = next(inferencer(image, return_vis=False, bbox_thr=bbox_score, nms_thr=0.5))
            tracker.add(frame, _detections(result, width, height, bbox_score))
            frame += 1
            source_frame += 1
    finally:
        capture.release()
    if frame < 1:
        raise RuntimeError("video has no decodable frames")
    keypoint = np.zeros((2, frame, 17, 2), dtype=np.float32)
    scores = np.zeros((2, frame, 17), dtype=np.float32)
    for person_index, track in enumerate(tracker.best_two()):
        for frame_index, (points, point_scores) in track.observations.items():
            keypoint[person_index, frame_index] = points
            scores[person_index, frame_index] = point_scores
    return {
        "schema_version": np.asarray("rtmpose_coco17_2d.v1"), "keypoint": keypoint,
        "keypoint_score": scores, "valid_mask": scores >= joint_score_threshold,
        "fps": np.asarray(fps * frame / max(source_frame, 1), dtype=np.float32),
        "source_fps": np.asarray(fps, dtype=np.float32), "total_frames": np.asarray(frame, dtype=np.int32),
        "width": np.asarray(width, dtype=np.int32), "height": np.asarray(height, dtype=np.int32),
    }


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.frame_stride < 1 or args.max_frames < 1 or not 0 <= args.joint_score_threshold <= 1:
        raise ValueError("invalid RTMPose17 extraction options")
    from mmpose.apis import MMPoseInferencer
    video = require_file(args.video, "input video")
    arrays = extract(video, MMPoseInferencer(pose2d=args.pose2d, device=args.device),
                     frame_stride=args.frame_stride, max_frames=args.max_frames,
                     bbox_score=args.bbox_score, joint_score_threshold=args.joint_score_threshold)
    output = Path(args.feature); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    log_event("rtmpose17_pose_complete", video=str(video), feature=str(output),
              frames=int(arrays["total_frames"]), valid_joints=int(arrays["valid_mask"].sum()))


if __name__ == "__main__":
    main()
