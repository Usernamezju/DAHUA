"""Reproducible semantic graph and measured-evidence generation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np


def build_semantic_graph(sample_id, track_ids, center, velocity, valid_mask, timestamps_s,
                         approach_threshold=0.0, contact_distance=0.75):
    centers = np.asarray(center, dtype=np.float32)
    velocities = np.asarray(velocity, dtype=np.float32)
    validity = np.asarray(valid_mask, dtype=bool)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    ids = np.asarray(track_ids, dtype=np.int64)
    if centers.shape != velocities.shape or centers.shape[-1] != 2:
        raise ValueError("center and velocity must have shape [M,T,2]")
    if validity.shape[:2] != centers.shape[:2] or len(timestamps) != centers.shape[1]:
        raise ValueError("semantic graph inputs have inconsistent shapes")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("semantic graph requires at least two increasing timestamps")
    persons = []
    for row, track_id in enumerate(ids):
        speed = np.linalg.norm(velocities[row], axis=-1)
        direction = np.arctan2(velocities[row, :, 1], velocities[row, :, 0])
        turns = int(np.sum(np.abs(np.diff(np.unwrap(direction))) > np.pi / 3))
        persons.append({
            "id": int(track_id), "mean_speed": float(speed.mean()), "turn_count": turns,
            "pose_coverage": float(validity[row].mean()),
        })
    relations = []
    segments = []
    for left, right in combinations(range(len(ids)), 2):
        delta = centers[right] - centers[left]
        distance = np.linalg.norm(delta, axis=-1)
        relative_velocity = velocities[right] - velocities[left]
        radial = np.sum(relative_velocity * delta, axis=-1) / np.maximum(distance, 1e-6)
        approaching = radial < approach_threshold
        contact = distance < contact_distance
        dt = np.diff(timestamps, append=timestamps[-1] + np.median(np.diff(timestamps)))
        for relation_type, mask in (("approach", approaching), ("possible_contact", contact)):
            duration_ms = int(round(float(dt[mask].sum()) * 1000))
            if duration_ms:
                relations.append({
                    "src": int(ids[left]), "dst": int(ids[right]), "type": relation_type,
                    "duration_ms": duration_ms, "confidence": float(mask.mean()),
                    "raw": {"minimum_distance": float(distance.min())},
                })
        motion = "close_contact" if contact.mean() >= 0.25 else "approaching" if approaching.mean() >= 0.5 else "other"
        segments.append({"id": f"s{len(segments)}", "start_ms": int(timestamps[0] * 1000),
                         "end_ms": int(timestamps[-1] * 1000), "motion": motion})
    return {
        "schema_version": "semantic_graph.v1", "sample_id": sample_id,
        "segments": segments, "persons": persons, "relations": relations,
        "quality": {"pose_coverage": float(validity.mean()) if validity.size else 0.0},
    }


def summarize_ntu25_pose_feature(sample_id: str, path: str | Path) -> dict:
    """Summarize an exported pose artifact using measured quantities only.

    The historical name is retained for backwards compatibility.  Besides the
    MediaPipe NTU-25 format this accepts the Campus6 RTMPose COCO-17 format;
    both expose people as ``[M,T,V]`` confidence/validity arrays.
    """
    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"pose feature does not exist: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as artifact:
        if "fps" not in artifact:
            raise ValueError("pose feature is missing fps")
        if "valid_mask" in artifact:
            valid = np.asarray(artifact["valid_mask"], dtype=bool)
        elif "keypoint_score" in artifact:
            valid = np.asarray(artifact["keypoint_score"], dtype=np.float32) >= 0.20
        else:
            raise ValueError("pose feature is missing valid_mask and keypoint_score")
        fps = float(np.asarray(artifact["fps"]).item())
        if "image_keypoint" in artifact:
            points = np.asarray(artifact["image_keypoint"], dtype=np.float32)
            width = max(1.0, float(np.asarray(artifact["width"]).item()))
            height = max(1.0, float(np.asarray(artifact["height"]).item()))
            points = points / np.asarray((width, height), dtype=np.float32)
        elif "keypoint" in artifact:
            points = np.asarray(artifact["keypoint"], dtype=np.float32)[..., :2]
        else:
            raise ValueError("pose feature is missing image_keypoint and keypoint")

    if valid.ndim != 3 or points.shape[:3] != valid.shape or points.shape[-1] != 2:
        raise ValueError("pose feature keypoints and valid_mask have incompatible shapes")
    if not np.isfinite(fps) or fps <= 0 or valid.shape[1] == 0:
        raise ValueError("pose feature has invalid timing")

    present = valid.any(axis=2)
    counts = valid.sum(axis=2).clip(min=1)[..., None]
    centers = (points * valid[..., None]).sum(axis=2) / counts
    centers[~present] = 0.0
    velocity = np.zeros_like(centers)
    if centers.shape[1] > 1:
        velocity[:, 1:] = np.diff(centers, axis=1) * fps
        velocity[:, 0] = velocity[:, 1]
    speed = np.linalg.norm(velocity, axis=2)

    persons = []
    for person_index in range(valid.shape[0]):
        observed_speed = speed[person_index, present[person_index]]
        persons.append(
            {
                "id": person_index,
                "pose_coverage": float(valid[person_index].mean()),
                "frame_coverage": float(present[person_index].mean()),
                "mean_normalized_speed": (
                    float(observed_speed.mean()) if observed_speed.size else 0.0
                ),
            }
        )

    relations = []
    if valid.shape[0] >= 2:
        both = present[0] & present[1]
        if both.any():
            distance = np.linalg.norm(centers[0] - centers[1], axis=1)
            observed = distance[both]
            relations.append(
                {
                    "src": 0,
                    "dst": 1,
                    "type": "measured_distance",
                    "minimum_normalized_distance": float(observed.min()),
                    "mean_normalized_distance": float(observed.mean()),
                    "frame_coverage": float(both.mean()),
                }
            )

    active_rows = present.any(axis=1)
    active_coverage = float(valid[active_rows].mean()) if active_rows.any() else 0.0
    duration_ms = int(round(valid.shape[1] / fps * 1000.0))
    return {
        "schema_version": "semantic_graph.v1",
        "sample_id": sample_id,
        "segments": [
            {
                "id": "s0",
                "start_ms": 0,
                "end_ms": max(1, duration_ms),
                "motion": "full_observed_clip",
            }
        ],
        "persons": persons,
        "relations": relations,
        "quality": {"pose_coverage": active_coverage},
    }
