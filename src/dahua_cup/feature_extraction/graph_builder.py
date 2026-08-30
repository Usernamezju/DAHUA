"""COCO17 to ProtoGCN coco_new conversion and normalized model views."""

from __future__ import annotations

import numpy as np


COCO17_JOINTS = 17
COCO20_JOINTS = 20


def build_coco20(
    keypoint: np.ndarray,
    keypoint_score: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append hip(17), torso(18), shoulder(19), matching Kinetics_Transform."""
    points = np.asarray(keypoint, dtype=np.float32)
    scores = np.asarray(keypoint_score, dtype=np.float32)
    masks = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 4 or points.shape[-2:] != (17, 2):
        raise ValueError("keypoint must have shape [M, T, 17, 2]")
    if scores.shape != points.shape[:-1] or masks.shape != scores.shape:
        raise ValueError("score and mask shapes must match keypoints")

    extra_points = np.zeros(points.shape[:-2] + (3, 2), dtype=np.float32)
    extra_scores = np.zeros(scores.shape[:-1] + (3,), dtype=np.float32)
    extra_masks = np.zeros(masks.shape[:-1] + (3,), dtype=bool)
    # Exact upstream node ordering: hip=17, torso=18, shoulder=19.
    extra_points[..., 0, :] = (points[..., 11, :] + points[..., 12, :]) / 2
    extra_points[..., 2, :] = (points[..., 5, :] + points[..., 6, :]) / 2
    extra_points[..., 1, :] = (extra_points[..., 0, :] + extra_points[..., 2, :]) / 2
    extra_scores[..., 0] = (scores[..., 11] + scores[..., 12]) / 2
    extra_scores[..., 2] = (scores[..., 5] + scores[..., 6]) / 2
    extra_scores[..., 1] = (extra_scores[..., 0] + extra_scores[..., 2]) / 2
    extra_masks[..., 0] = masks[..., 11] & masks[..., 12]
    extra_masks[..., 2] = masks[..., 5] & masks[..., 6]
    extra_masks[..., 1] = extra_masks[..., 0] & extra_masks[..., 2]
    extra_scores *= extra_masks
    return (
        np.concatenate((points, extra_points), axis=-2),
        np.concatenate((scores, extra_scores), axis=-1),
        np.concatenate((masks, extra_masks), axis=-1),
    )


def build_protogcn_views(
    keypoint: np.ndarray,
    keypoint_score: np.ndarray,
    valid_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    """Return image-normalized and body-centered [M,T,20,3] inputs."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    points, scores, masks = build_coco20(keypoint, keypoint_score, valid_mask)
    image_points = points.copy()
    image_points[..., 0] = image_points[..., 0] / image_width * 2.0 - 1.0
    image_points[..., 1] = image_points[..., 1] / image_height * 2.0 - 1.0

    centered = points - points[..., 17:18, :]
    torso_scale = np.linalg.norm(points[..., 19, :] - points[..., 17, :], axis=-1)
    torso_scale = np.maximum(torso_scale, 1e-6)
    centered /= torso_scale[..., None, None]
    confidence = np.where(masks, scores, 0.0)[..., None]
    return {
        "image": np.concatenate((image_points, confidence), axis=-1).astype(np.float32),
        "body": np.concatenate((centered, confidence), axis=-1).astype(np.float32),
        "valid_mask": masks,
    }
