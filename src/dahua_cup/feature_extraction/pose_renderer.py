"""Privacy-first COCO skeleton renderer with no RGB dependency."""

from __future__ import annotations

import numpy as np


COCO_EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
              (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
              (1, 0), (3, 1), (2, 0), (4, 2))


def render_pose_frame(keypoint, valid_mask, width, height, track_ids=None):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("pose rendering requires opencv-python") from exc
    points = np.asarray(keypoint, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or points.shape[-2:] != (17, 2) or valid.shape != points.shape[:-1]:
        raise ValueError("expected keypoint [M,17,2] and valid_mask [M,17]")
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    colors = ((75, 180, 255), (255, 120, 120), (120, 255, 160), (220, 160, 255))
    for person, joints in enumerate(points):
        color = colors[person % len(colors)]
        for first, second in COCO_EDGES:
            if valid[person, first] and valid[person, second]:
                cv2.line(canvas, tuple(np.rint(joints[first]).astype(int)),
                         tuple(np.rint(joints[second]).astype(int)), color, 2, cv2.LINE_AA)
        for joint, is_valid in zip(joints, valid[person]):
            if is_valid:
                cv2.circle(canvas, tuple(np.rint(joint).astype(int)), 3, color, -1, cv2.LINE_AA)
        if track_ids is not None and valid[person].any():
            anchor = tuple(np.rint(joints[np.flatnonzero(valid[person])[0]]).astype(int))
            cv2.putText(canvas, f"ID {int(track_ids[person])}", anchor, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return canvas
