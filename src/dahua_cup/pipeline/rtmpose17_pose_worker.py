"""Extract a single video as the Campus6 direct RTMPose COCO-17 artifact."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional

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
    value.add_argument("--pose2d-weights")
    value.add_argument("--det-weights")
    value.add_argument(
        "--extractor-id", default="rtm-s-coco17.v1",
        help="provenance recorded in the derived COCO-17 artifact",
    )
    value.add_argument("--backend", choices=("mmpose", "mmdeploy"), default="mmpose")
    value.add_argument("--detector-model-dir")
    value.add_argument("--pose-model-dir")
    return value


def _checkpoint_config(path: Path, name: str):
    """Load the exact expanded MMEngine config embedded in a project weight.

    Reconstructing from checkpoint metadata prevents a silent mismatch between
    an RTMDet head and its configuration. The shipped RTMDet-S is the official
    COCO-80 checkpoint (person is category 0); RTMPose-S emits the 17 COCO
    joints consumed by Campus6. Both weights contain this metadata.
    """
    import torch
    from mmengine.config import Config

    payload = torch.load(path, map_location="cpu")
    config_text = (payload.get("meta") or {}).get("cfg")
    if not isinstance(config_text, str) or not config_text.strip():
        raise ValueError(f"{name} checkpoint has no embedded MMEngine config")
    return Config.fromstring(config_text, file_format=".py")


def _write_checkpoint_config(path: Path, name: str, directory: Path) -> Path:
    """Materialize the checkpoint's expanded config for MMDetection.

    ``Pose2DInferencer`` accepts a Config object, but its detector helper in
    MMPose 1.3 expects a config *path*.  The temporary file lives only while
    the inferencers are built; both models retain the resolved config after
    construction.
    """
    output = directory / f"{name}.py"
    output.write_text(_checkpoint_config(path, name).pretty_text, encoding="utf-8")
    return output


def _installed_s_config(component: str, checkpoint: Path) -> Optional[Path]:
    """Return the ABI-compatible MMPose/MMDet config for shipped S weights.

    The January 2023 RTMPose-S checkpoint embeds the legacy ``RTMHead`` name.
    MMPose 1.3 registers its compatible successor as ``RTMCCHead``.  Its
    packaged S COCO config is state-dict compatible, whereas blindly loading
    the legacy metadata fails before weights can be restored.  Old, custom M
    assets continue through the embedded-config path below.
    """
    name = checkpoint.name.lower()
    if component == "pose" and name == "rtmpose-s_coco17.pth":
        import mmpose

        path = (Path(mmpose.__file__).resolve().parent / ".mim" / "configs" /
                "body_2d_keypoint" / "rtmpose" / "coco" /
                "rtmpose-s_8xb256-420e_coco-256x192.py")
    elif component == "detector" and name == "rtmdet-s_coco80.pth":
        import mmdet

        path = (Path(mmdet.__file__).resolve().parent / ".mim" / "configs" /
                "rtmdet" / "rtmdet_s_8xb32-300e_coco.py")
    else:
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"installed {component} package lacks the required RTM-S config: {path}"
        )
    return path


def create_mmpose_inferencer(args):
    """Build an offline MMPose inferencer when local shipped weights exist."""
    from mmpose.apis import MMPoseInferencer

    if not args.pose2d_weights and not args.det_weights:
        return MMPoseInferencer(pose2d=args.pose2d, device=args.device)
    if not args.pose2d_weights or not args.det_weights:
        raise ValueError(
            "--pose2d-weights and --det-weights must be supplied together"
        )
    pose_weights = require_file(args.pose2d_weights, "RTMPose COCO-17 weights")
    detector_weights = require_file(args.det_weights, "RTMDet person weights")
    # MMPoseInferencer publicly documents a config path for ``pose2d`` but
    # delegates to Pose2DInferencer, whose stable ModelType accepts Config.
    # Calling it directly is necessary for our embedded one-class detector
    # config and preserves the same callable result contract used by extract.
    from mmpose.apis.inferencers import Pose2DInferencer
    with tempfile.TemporaryDirectory(prefix="dahua_pose_configs_") as root:
        root_path = Path(root)
        pose_config = _installed_s_config("pose", pose_weights)
        detector_config = _installed_s_config("detector", detector_weights)
        return Pose2DInferencer(
            model=(str(pose_config) if pose_config else _checkpoint_config(pose_weights, "RTMPose")),
            weights=str(pose_weights),
            det_model=(str(detector_config) if detector_config else str(
                _write_checkpoint_config(detector_weights, "rtmdet_person", root_path)
            )),
            det_weights=str(detector_weights),
            det_cat_ids=[0],
            device=args.device,
        )


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


def _extract_video(video: Path, predict, *, frame_stride: int, max_frames: int,
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
            tracker.add(frame, predict(image, width, height))
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


def extract(video: Path, inferencer, *, frame_stride: int, max_frames: int, bbox_score: float,
            joint_score_threshold: float) -> dict:
    def predict(image, width: int, height: int):
        result = next(inferencer(image, return_vis=False, bbox_thr=bbox_score, nms_thr=0.5))
        return _detections(result, width, height, bbox_score)

    return _extract_video(
        video, predict, frame_stride=frame_stride, max_frames=max_frames,
        joint_score_threshold=joint_score_threshold,
    )


def _device_parts(device: str) -> tuple[str, int]:
    kind, _, index = device.partition(":")
    if kind not in {"cpu", "cuda"}:
        raise ValueError("MMDeploy backend only supports cpu or cuda devices")
    return kind, int(index or 0)


def _nms_keep(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apply the same score-filtered greedy NMS as MMPose's top-down path."""
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


def _mmdeploy_detections(result, keypoints, width: int, height: int, threshold: float):
    """Convert MMDeploy Detector/PoseDetector results to the worker contract."""
    bboxes, labels, *_ = result
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, np.asarray(bboxes).shape[-1])
    labels = np.asarray(labels).reshape(-1)
    values = np.asarray(keypoints, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    parsed = []
    for index, box in enumerate(boxes):
        if index >= values.shape[0] or labels[index] != 0 or box.shape[0] < 5:
            continue
        score = float(box[4])
        points = values[index, :, :2].copy()
        scores = values[index, :, 2] if values.shape[-1] >= 3 else np.ones(points.shape[0], dtype=np.float32)
        if points.shape != (17, 2) or scores.shape != (17,) or score < threshold:
            continue
        points[:, 0] /= max(width, 1)
        points[:, 1] /= max(height, 1)
        parsed.append((box[:4], points, scores, score))
    return parsed


def extract_mmdeploy(video: Path, detector_model_dir: Path, pose_model_dir: Path, *, device: str,
                     frame_stride: int, max_frames: int, bbox_score: float,
                     joint_score_threshold: float) -> dict:
    try:
        from mmdeploy_runtime import Detector, PoseDetector
    except ImportError as exc:
        raise RuntimeError("MMDeploy runtime is not installed; install requirements.txt after TensorRT setup") from exc
    kind, device_id = _device_parts(device)
    detector = Detector(str(detector_model_dir), device_name=kind, device_id=device_id)
    pose_detector = PoseDetector(str(pose_model_dir), device_name=kind, device_id=device_id)

    def predict(image, width: int, height: int):
        detection = detector(image)
        boxes = np.asarray(detection[0], dtype=np.float32)
        if boxes.size == 0:
            return []
        labels = np.asarray(detection[1]).reshape(-1)
        person_boxes = boxes[labels == 0]
        person_boxes = person_boxes[person_boxes[:, 4] > bbox_score]
        person_boxes = person_boxes[_nms_keep(person_boxes, 0.5)]
        if person_boxes.size == 0:
            return []
        pose = pose_detector(image, person_boxes[:, :4])
        filtered_detection = (person_boxes, np.zeros((len(person_boxes),), dtype=np.int64))
        return _mmdeploy_detections(filtered_detection, pose, width, height, bbox_score)

    return _extract_video(
        video, predict, frame_stride=frame_stride, max_frames=max_frames,
        joint_score_threshold=joint_score_threshold,
    )


def main(argv=None) -> None:
    total_started = time.perf_counter()
    args = parser().parse_args(argv)
    if args.frame_stride < 1 or args.max_frames < 1 or not 0 <= args.joint_score_threshold <= 1:
        raise ValueError("invalid RTMPose17 extraction options")
    video = require_file(args.video, "input video")
    if args.backend == "mmdeploy":
        if not args.detector_model_dir or not args.pose_model_dir:
            raise ValueError("MMDeploy backend requires --detector-model-dir and --pose-model-dir")
        detector_model_dir = Path(args.detector_model_dir)
        pose_model_dir = Path(args.pose_model_dir)
        if not detector_model_dir.is_dir() or not pose_model_dir.is_dir():
            raise FileNotFoundError("MMDeploy detector and pose model directories must exist")
        arrays = extract_mmdeploy(
            video, detector_model_dir, pose_model_dir, device=args.device,
            frame_stride=args.frame_stride, max_frames=args.max_frames,
            bbox_score=args.bbox_score, joint_score_threshold=args.joint_score_threshold,
        )
    else:
        arrays = extract(video, create_mmpose_inferencer(args),
                         frame_stride=args.frame_stride, max_frames=args.max_frames,
                         bbox_score=args.bbox_score, joint_score_threshold=args.joint_score_threshold)
    total_seconds = time.perf_counter() - total_started
    arrays["extractor_id"] = np.asarray(args.extractor_id)
    arrays["extract_elapsed_seconds"] = np.asarray(round(total_seconds, 4), dtype=np.float32)
    output = Path(args.feature); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    log_event("rtmpose17_pose_complete", video=str(video), feature=str(output),
              frames=int(arrays["total_frames"]), valid_joints=int(arrays["valid_mask"].sum()),
              extractor_id=args.extractor_id, elapsed_seconds=round(total_seconds, 4))


if __name__ == "__main__":
    main()
