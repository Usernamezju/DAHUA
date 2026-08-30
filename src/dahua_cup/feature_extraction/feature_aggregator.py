"""Aggregate tracked COCO poses into the feature.v1 tensor contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schemas import QualityMetrics, validate_feature_arrays


class FeatureAggregator:
    """Build dense per-track tensors while retaining missing-point masks."""

    def __init__(self, keypoint_score_threshold: float = 0.2, max_people: int = 2):
        if not 0 <= keypoint_score_threshold <= 1 or max_people < 1:
            raise ValueError("invalid aggregator parameters")
        self.score_threshold = float(keypoint_score_threshold)
        self.max_people = int(max_people)

    def aggregate(
        self,
        frames: Sequence[Sequence[Mapping]],
        timestamps_s: Iterable[float],
    ) -> tuple[dict[str, np.ndarray], QualityMetrics]:
        timestamps = np.asarray(list(timestamps_s), dtype=np.float64)
        if len(frames) != len(timestamps) or len(frames) == 0:
            raise ValueError("frames and non-empty timestamps must have equal length")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")

        by_id = defaultdict(list)
        for frame_index, people in enumerate(frames):
            for person in people:
                by_id[int(person["track_id"])].append((frame_index, person))
        ranked = sorted(by_id, key=lambda track_id: (-len(by_id[track_id]), track_id))
        track_ids = ranked[: self.max_people]
        m, t = len(track_ids), len(frames)
        keypoint = np.zeros((m, t, 17, 2), dtype=np.float32)
        scores = np.zeros((m, t, 17), dtype=np.float32)
        bbox = np.zeros((m, t, 4), dtype=np.float32)

        for row, track_id in enumerate(track_ids):
            for frame_index, person in by_id[track_id]:
                points = np.asarray(person["keypoint"], dtype=np.float32)
                point_scores = np.asarray(person["keypoint_score"], dtype=np.float32)
                box = np.asarray(person["bbox"], dtype=np.float32)
                if points.shape != (17, 2) or point_scores.shape != (17,) or box.shape != (4,):
                    raise ValueError("each pose must contain COCO17 keypoints, scores and xyxy bbox")
                keypoint[row, frame_index] = points
                scores[row, frame_index] = np.clip(point_scores, 0.0, 1.0)
                bbox[row, frame_index] = box

        valid = scores >= self.score_threshold
        center = self._centers(keypoint, bbox, valid)
        velocity = self._gradient(center, timestamps)
        acceleration = self._gradient(velocity, timestamps)
        arrays = validate_feature_arrays({
            "keypoint": keypoint,
            "keypoint_score": scores,
            "bbox": bbox,
            "track_id": np.asarray(track_ids, dtype=np.int64),
            "valid_mask": valid,
            "center": center,
            "velocity": velocity,
            "acceleration": acceleration,
        })
        present = np.any(scores > 0, axis=2)
        gaps = sum(max(0, self._runs(~row) - 1) for row in present) if m else 0
        quality = QualityMetrics(
            pose_coverage=float(valid.mean()) if valid.size else 0.0,
            mean_joint_score=float(scores[valid].mean()) if valid.any() else 0.0,
            track_fragmentation=min(1.0, gaps / max(1, m * t)),
            id_switch_estimate=0,
        )
        return arrays, quality

    @staticmethod
    def _centers(keypoint: np.ndarray, bbox: np.ndarray, valid: np.ndarray) -> np.ndarray:
        hip_valid = valid[..., 11] & valid[..., 12]
        hip_center = (keypoint[..., 11, :] + keypoint[..., 12, :]) / 2.0
        bbox_center = (bbox[..., :2] + bbox[..., 2:]) / 2.0
        return np.where(hip_valid[..., None], hip_center, bbox_center).astype(np.float32)

    @staticmethod
    def _gradient(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values)
        if values.shape[1] > 1:
            dt = np.diff(timestamps).astype(np.float32)
            result[:, 1:] = np.diff(values, axis=1) / dt[None, :, None]
            result[:, 0] = result[:, 1]
        return result

    @staticmethod
    def _runs(mask: np.ndarray) -> int:
        padded = np.pad(mask.astype(np.int8), (1, 1))
        return int(np.sum(np.diff(padded) == 1))
