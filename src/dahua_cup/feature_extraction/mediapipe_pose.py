"""MediaPipe Pose Landmarker helpers for ProtoGCN NTU-25 input."""

from __future__ import annotations

from itertools import permutations

import numpy as np


NUM_MEDIAPIPE_LANDMARKS = 33
NUM_NTU_JOINTS = 25

# Each NTU joint is either copied from one MediaPipe landmark or averaged from
# several landmarks. NTU indices follow the zero-based NTU RGB+D 25-joint
# layout used by ProtoGCN.
NTU25_FROM_MEDIAPIPE33 = (
    (23, 24),          # 0: spine base
    (23, 24, 11, 12), # 1: spine mid
    (11, 12),          # 2: neck
    (0, 7, 8),         # 3: head
    (11,),             # 4: left shoulder
    (13,),             # 5: left elbow
    (15,),             # 6: left wrist
    (15, 17, 19, 21), # 7: left hand
    (12,),             # 8: right shoulder
    (14,),             # 9: right elbow
    (16,),             # 10: right wrist
    (16, 18, 20, 22), # 11: right hand
    (23,),             # 12: left hip
    (25,),             # 13: left knee
    (27,),             # 14: left ankle
    (31,),             # 15: left foot
    (24,),             # 16: right hip
    (26,),             # 17: right knee
    (28,),             # 18: right ankle
    (32,),             # 19: right foot
    (11, 12),          # 20: spine shoulder
    (19,),             # 21: left hand tip
    (21,),             # 22: left thumb
    (20,),             # 23: right hand tip
    (22,),             # 24: right thumb
)


def mediapipe33_to_ntu25(landmarks, visibility=None, score_threshold=0.2):
    """Map one MediaPipe 33-landmark 3D pose to NTU RGB+D 25 joints.

    Missing/low-confidence joints are zeroed because ProtoGCN's
    ``PreNormalize3D`` transform uses all-zero coordinates as its missing-joint
    convention.
    """

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (NUM_MEDIAPIPE_LANDMARKS, 3):
        raise ValueError(
            f"MediaPipe landmarks must have shape (33, 3), got {points.shape}"
        )

    if visibility is None:
        scores = np.ones(NUM_MEDIAPIPE_LANDMARKS, dtype=np.float32)
    else:
        scores = np.asarray(visibility, dtype=np.float32)
        if scores.shape != (NUM_MEDIAPIPE_LANDMARKS,):
            raise ValueError(
                f"MediaPipe visibility must have shape (33,), got {scores.shape}"
            )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)

    mapped = np.zeros((NUM_NTU_JOINTS, 3), dtype=np.float32)
    mapped_scores = np.zeros(NUM_NTU_JOINTS, dtype=np.float32)
    for ntu_index, source_indices in enumerate(NTU25_FROM_MEDIAPIPE33):
        mapped[ntu_index] = points[list(source_indices)].mean(axis=0)
        mapped_scores[ntu_index] = scores[list(source_indices)].mean()

    finite = np.isfinite(mapped).all(axis=1)
    valid = finite & (mapped_scores >= float(score_threshold))
    mapped[~valid] = 0.0
    return mapped, mapped_scores, valid


def normalized_pose_center(landmarks):
    """Return the normalized-image hip center used for two-person association."""

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape[0] != NUM_MEDIAPIPE_LANDMARKS or points.shape[1] < 2:
        raise ValueError("normalized MediaPipe landmarks must have shape (33, >=2)")
    return points[[23, 24], :2].mean(axis=0)


def assign_pose_slots(centers, previous_centers=None, max_people=2):
    """Assign current detections to stable person slots.

    MediaPipe returns at most a few poses, so exhaustive assignment is clearer
    and deterministic. On the first frame, people are ordered left-to-right.
    Later frames minimize normalized hip-center displacement.
    """

    current = np.asarray(centers, dtype=np.float32)
    if current.size == 0:
        return []
    if current.ndim != 2 or current.shape[1] != 2:
        raise ValueError(f"pose centers must have shape (N, 2), got {current.shape}")
    if len(current) > max_people:
        raise ValueError(f"received {len(current)} poses, max_people={max_people}")

    previous = [None] * max_people if previous_centers is None else list(previous_centers)
    if len(previous) != max_people:
        raise ValueError("previous_centers length must equal max_people")

    occupied = [index for index, value in enumerate(previous) if value is not None]
    if not occupied:
        order = np.argsort(current[:, 0], kind="stable")
        slots = [0] * len(current)
        for slot, detection_index in enumerate(order):
            slots[int(detection_index)] = slot
        return slots

    best_slots = None
    best_cost = float("inf")
    for candidate in permutations(range(max_people), len(current)):
        cost = 0.0
        for detection_index, slot in enumerate(candidate):
            if previous[slot] is None:
                cost += 1.0
            else:
                delta = current[detection_index] - np.asarray(previous[slot])
                cost += float(np.linalg.norm(delta))
        if cost < best_cost:
            best_cost = cost
            best_slots = candidate
    return list(best_slots)

