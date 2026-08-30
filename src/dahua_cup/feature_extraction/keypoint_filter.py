"""One Euro Filter smoothing for tracked COCO keypoints."""

import math

import numpy as np


class OneEuroFilter:
    """Vectorized One Euro Filter with optional validity masking."""

    def __init__(self, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0):
        if min_cutoff <= 0 or derivative_cutoff <= 0 or beta < 0:
            raise ValueError("One Euro parameters must satisfy cutoff > 0 and beta >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.timestamp = None
        self.raw_previous = None
        self.filtered_previous = None
        self.derivative_previous = None
        self.initialized = None

    @staticmethod
    def _alpha(cutoff, elapsed):
        time_constant = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + time_constant / elapsed)

    def update(self, value, timestamp, valid=None):
        value = np.asarray(value, dtype=np.float32)
        valid = np.ones(value.shape, dtype=bool) if valid is None else np.broadcast_to(valid, value.shape)
        if self.timestamp is None:
            self.timestamp = float(timestamp)
            self.raw_previous = value.copy()
            self.filtered_previous = value.copy()
            self.derivative_previous = np.zeros_like(value)
            self.initialized = valid.copy()
            return value.copy()

        elapsed = max(float(timestamp) - self.timestamp, 1e-6)
        output = value.copy()
        newly_valid = valid & ~self.initialized
        continuing = valid & self.initialized

        derivative = (value - self.raw_previous) / elapsed
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        derivative_filtered = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.derivative_previous
        )
        cutoff = self.min_cutoff + self.beta * np.abs(derivative_filtered)
        value_alpha = self._alpha(cutoff, elapsed)
        smoothed = (
            value_alpha * value
            + (1.0 - value_alpha) * self.filtered_previous
        )

        output[continuing] = smoothed[continuing]
        self.raw_previous[valid] = value[valid]
        self.filtered_previous[continuing] = output[continuing]
        self.filtered_previous[newly_valid] = value[newly_valid]
        self.derivative_previous[continuing] = derivative_filtered[continuing]
        self.derivative_previous[newly_valid] = 0.0
        self.initialized |= valid
        self.timestamp = float(timestamp)
        return output


class TrackedKeypointFilter:
    """Maintain one independent One Euro Filter for every tracked identity."""

    def __init__(self, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0,
                 keypoint_score_threshold=0.2, max_idle_seconds=2.0):
        self.parameters = (min_cutoff, beta, derivative_cutoff)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.max_idle_seconds = float(max_idle_seconds)
        self.filters = {}
        self.last_seen = {}

    def update(self, poses, timestamp):
        """Smooth pose coordinates independently for each ``track_id``."""
        active_ids = set()
        filtered_poses = []
        for pose in poses:
            track_id = int(pose["track_id"])
            active_ids.add(track_id)
            if track_id not in self.filters:
                self.filters[track_id] = OneEuroFilter(*self.parameters)
            scores = np.asarray(pose["keypoint_score"], dtype=np.float32)
            valid = (scores >= self.keypoint_score_threshold)[:, None]
            filtered = dict(pose)
            filtered["keypoint"] = self.filters[track_id].update(
                pose["keypoint"], timestamp, valid=valid
            )
            filtered_poses.append(filtered)
            self.last_seen[track_id] = float(timestamp)

        stale_ids = [
            track_id for track_id, last_seen in self.last_seen.items()
            if track_id not in active_ids
            and float(timestamp) - last_seen > self.max_idle_seconds
        ]
        for track_id in stale_ids:
            del self.filters[track_id]
            del self.last_seen[track_id]
        return filtered_poses

    __call__ = update


def interpolate_short_gaps(keypoint, valid_mask, max_gap=5):
    """Linearly fill bounded gaps while preserving long-gap invalid masks.

    Args:
        keypoint: ``[T, V, 2]`` coordinates.
        valid_mask: ``[T, V]`` point validity.
        max_gap: Maximum number of missing frames to fill.
    """
    points = np.asarray(keypoint, dtype=np.float32).copy()
    valid = np.asarray(valid_mask, dtype=bool).copy()
    if points.ndim != 3 or points.shape[-1] != 2 or valid.shape != points.shape[:-1]:
        raise ValueError("expected keypoint [T,V,2] and valid_mask [T,V]")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    for joint in range(points.shape[1]):
        known = np.flatnonzero(valid[:, joint])
        for left, right in zip(known[:-1], known[1:]):
            gap = int(right - left - 1)
            if 0 < gap <= max_gap:
                fractions = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
                points[left + 1:right, joint] = (
                    points[left, joint][None] * (1 - fractions[:, None])
                    + points[right, joint][None] * fractions[:, None]
                )
                valid[left + 1:right, joint] = True
    return points, valid
