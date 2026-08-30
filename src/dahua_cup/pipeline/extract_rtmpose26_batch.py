"""Resumable top-down RTMPose Halpe-26 feature extraction.

The output keeps only the two most persistent detected people per video.  It
stores the canonical raw layout ``pose[T, M, V, C]`` where M=2, V=26 and
C=(normalised_x, normalised_y, keypoint_confidence).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        - overlap
    )
    return float(overlap / union) if union > 0 else 0.0


def _centre_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_centre = (left[:2] + left[2:]) / 2
    right_centre = (right[:2] + right[2:]) / 2
    scale = max(np.linalg.norm(left[2:] - left[:2]), np.linalg.norm(right[2:] - right[:2]), 1.0)
    return float(np.linalg.norm(left_centre - right_centre) / scale)


@dataclass
class Track:
    bbox: np.ndarray
    observations: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    last_frame: int = 0

    def update(self, frame: int, bbox: np.ndarray, point: np.ndarray, score: np.ndarray, det_score: float) -> None:
        self.bbox = bbox
        self.observations[frame] = (point, score)
        self.scores.append(det_score)
        self.areas.append(float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])))
        self.last_frame = frame

    def rank(self) -> float:
        # Reward persistence first, then prefer the main foreground people.
        return len(self.observations) * float(np.mean(self.scores)) * (1.0 + np.sqrt(np.median(self.areas)) / 1000.0)


class GreedyTracker:
    def __init__(self, max_gap: int = 12) -> None:
        self.tracks: List[Track] = []
        self.max_gap = max_gap

    def add(self, frame: int, detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> None:
        candidates = []
        for track_index, track in enumerate(self.tracks):
            if frame - track.last_frame > self.max_gap:
                continue
            for detection_index, (bbox, _, _, _) in enumerate(detections):
                iou = _iou(track.bbox, bbox)
                distance = _centre_distance(track.bbox, bbox)
                if iou >= 0.05 or distance <= 0.55:
                    candidates.append((iou - 0.15 * distance, track_index, detection_index))
        used_tracks, used_detections = set(), set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            bbox, point, score, det_score = detections[detection_index]
            self.tracks[track_index].update(frame, bbox, point, score, det_score)
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        for detection_index, (bbox, point, score, det_score) in enumerate(detections):
            if detection_index in used_detections:
                continue
            track = Track(bbox=bbox.copy(), last_frame=frame)
            track.update(frame, bbox, point, score, det_score)
            self.tracks.append(track)

    def best_two(self) -> List[Track]:
        return sorted(self.tracks, key=lambda track: track.rank(), reverse=True)[:2]


def _parse_predictions(result: Dict[str, Any], width: int, height: int, min_score: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    predictions = result.get("predictions", [])
    people = predictions[0] if predictions else []
    parsed = []
    for person in people:
        bbox = np.asarray(person.get("bbox", [[0, 0, 0, 0]]), dtype=np.float32).reshape(-1, 4)[0]
        det_score = float(np.asarray(person.get("bbox_score", [1.0])).reshape(-1)[0])
        points = np.asarray(person.get("keypoints", []), dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32).reshape(-1)
        if points.shape != (26, 2) or scores.shape != (26,) or det_score < min_score:
            continue
        points[:, 0] /= max(width, 1)
        points[:, 1] /= max(height, 1)
        parsed.append((bbox, points, scores, det_score))
    return parsed


def extract(
    video: Path,
    inferencer: Any,
    frame_stride: int,
    bbox_score: float,
    max_frames: int,
) -> Dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    source_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # ProtoGCN's official Kinetics pipeline uniformly samples 100 frames.  For
    # longer inputs, infer only those evenly spread frames instead of wasting
    # pose-model calls on frames the training pipeline would discard anyway.
    selected = None
    if max_frames > 0 and source_total > max_frames:
        selected = set(np.linspace(0, source_total - 1, max_frames, dtype=np.int32).tolist())
    tracker = GreedyTracker(max_gap=6 * frame_stride)
    sample_count, source_frame = 0, 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_frame % frame_stride or (selected is not None and source_frame not in selected):
                source_frame += 1
                continue
            height, width = bgr.shape[:2]
            result = next(inferencer(
                bgr,
                return_vis=False,
                bbox_thr=bbox_score,
                nms_thr=0.5,
            ))
            tracker.add(sample_count, _parse_predictions(result, width, height, bbox_score))
            sample_count += 1
            source_frame += 1
    finally:
        capture.release()
    if not sample_count:
        raise RuntimeError(f"video has no decodable frames: {video}")

    pose = np.zeros((sample_count, 2, 26, 3), dtype=np.float32)
    for person_index, track in enumerate(tracker.best_two()):
        for frame, (points, scores) in track.observations.items():
            pose[frame, person_index, :, :2] = points
            pose[frame, person_index, :, 2] = scores
    return {
        "schema_version": np.asarray("rtmpose_halpe26_topdown.v1"),
        "pose": pose,
        "keypoint": pose[..., :2].transpose(1, 0, 2, 3),
        "keypoint_score": pose[..., 2].transpose(1, 0, 2),
        "fps": np.asarray(fps * sample_count / max(source_frame, 1), dtype=np.float32),
        "source_fps": np.asarray(fps, dtype=np.float32),
        "source_total_frames": np.asarray(source_frame, dtype=np.int32),
        "frame_stride": np.asarray(frame_stride, dtype=np.int32),
        "total_frames": np.asarray(sample_count, dtype=np.int32),
        "max_people": np.asarray(2, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=100,
                        help="uniformly infer at most this many source frames; 0 keeps all frames")
    parser.add_argument("--bbox-score", type=float, default=0.15)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.worker_index < 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("invalid shard or frame stride")

    from mmpose.apis import MMPoseInferencer

    # ``body26`` is the official RTMPose-m Halpe-26 model alias.  Its default
    # person detector is RTMDet-m, yielding the intended top-down pipeline.
    inferencer = MMPoseInferencer(pose2d="body26", device=args.device)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    log_path = args.output_dir / "logs" / f"worker_{args.worker_index:02d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if log_path.exists():
        done = {json.loads(line).get("video") for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    counts = {"ok": 0, "error": 0, "skip": 0}
    with log_path.open("a", encoding="utf-8") as log:
        for number, row in enumerate(rows, 1):
            video = Path(row["video"])
            digest = hashlib.sha1(str(video).encode()).hexdigest()[:12]
            output = args.output_dir / row["label"] / f"{video.stem}__{digest}.npz"
            if str(video) in done and output.is_file():
                counts["skip"] += 1
                continue
            try:
                arrays = extract(video, inferencer, args.frame_stride, args.bbox_score, args.max_frames)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **arrays)
                event = {"status": "ok", "video": str(video), "label": row["label"], "feature": str(output)}
                counts["ok"] += 1
            except Exception as error:
                event = {"status": "error", "video": str(video), "label": row["label"], "error": f"{type(error).__name__}: {error}"}
                counts["error"] += 1
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            if number % 5 == 0 or number == len(rows):
                print(f"[{number}/{len(rows)}] {counts}", flush=True)


if __name__ == "__main__":
    main()
