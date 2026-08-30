"""Deterministic zero/one/two-person selection and group cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    """One person detection in source-image pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float

    def as_dict(self) -> dict:
        return {
            "bbox_xyxy": [float(value) for value in self.bbox_xyxy],
            "confidence": float(self.confidence),
        }


def _valid_detection(
    detection: PersonDetection,
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    minimum_area: float,
) -> PersonDetection | None:
    if not np.isfinite(detection.confidence):
        return None
    if detection.confidence < confidence_threshold:
        return None
    values = np.asarray(detection.bbox_xyxy, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    left = float(np.clip(values[0], 0.0, width))
    top = float(np.clip(values[1], 0.0, height))
    right = float(np.clip(values[2], 0.0, width))
    bottom = float(np.clip(values[3], 0.0, height))
    if right <= left or bottom <= top:
        return None
    if (right - left) * (bottom - top) < minimum_area:
        return None
    return PersonDetection(
        bbox_xyxy=(left, top, right, bottom),
        confidence=float(detection.confidence),
    )


def select_people(
    detections: Iterable[PersonDetection],
    *,
    width: int,
    height: int,
    confidence_threshold: float = 0.35,
    minimum_area_fraction: float = 0.0001,
    maximum_people: int = 2,
) -> tuple[list[PersonDetection], int]:
    """Filter detections and retain the most confident one or two people.

    Returns the selected detections and the number of additional valid people.
    The overflow count makes a crowd-capacity violation explicit instead of
    silently claiming that the selected pair represents everyone in the frame.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if maximum_people not in (1, 2):
        raise ValueError("maximum_people must be one or two")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= minimum_area_fraction <= 1.0:
        raise ValueError("minimum_area_fraction must be in [0, 1]")
    minimum_area = width * height * minimum_area_fraction
    valid = [
        checked
        for detection in detections
        if (
            checked := _valid_detection(
                detection,
                width=width,
                height=height,
                confidence_threshold=confidence_threshold,
                minimum_area=minimum_area,
            )
        )
        is not None
    ]
    valid.sort(key=lambda item: item.confidence, reverse=True)
    return valid[:maximum_people], max(0, len(valid) - maximum_people)


def build_group_crop(
    people: Sequence[PersonDetection],
    *,
    width: int,
    height: int,
    context_factor: float = 1.20,
    minimum_crop_fraction: float = 0.20,
) -> tuple[float, float, float, float]:
    """Return a deterministic group crop that contains zero, one or two people.

    The crop keeps the source aspect ratio unconstrained. The video worker later
    letterboxes this rectangle to its output size, so people are never distorted.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(people) > 2:
        raise ValueError("group crop accepts at most two selected people")
    if context_factor < 1.0:
        raise ValueError("context_factor must be at least one")
    if not 0.0 < minimum_crop_fraction <= 1.0:
        raise ValueError("minimum_crop_fraction must be in (0, 1]")
    if not people:
        return 0.0, 0.0, float(width), float(height)

    boxes = np.asarray(
        [person.bbox_xyxy for person in people], dtype=np.float32
    )
    union = np.asarray(
        (
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].max(),
            boxes[:, 3].max(),
        ),
        dtype=np.float32,
    )
    center_x = float((union[0] + union[2]) / 2.0)
    center_y = float((union[1] + union[3]) / 2.0)
    crop_width = max(
        float(union[2] - union[0]) * context_factor,
        width * minimum_crop_fraction,
    )
    crop_height = max(
        float(union[3] - union[1]) * context_factor,
        height * minimum_crop_fraction,
    )
    crop_width = min(float(width), crop_width)
    crop_height = min(float(height), crop_height)
    left = min(max(0.0, center_x - crop_width / 2.0), width - crop_width)
    top = min(max(0.0, center_y - crop_height / 2.0), height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    # Clamping a crop near an image edge must never exclude a selected person.
    left = min(left, float(union[0]))
    top = min(top, float(union[1]))
    right = max(right, float(union[2]))
    bottom = max(bottom, float(union[3]))
    return (
        max(0.0, left),
        max(0.0, top),
        min(float(width), right),
        min(float(height), bottom),
    )


def crop_and_letterbox(
    frame: np.ndarray,
    crop_xyxy: Sequence[float],
    output_size: int,
    *,
    padding_value: int = 114,
) -> tuple[np.ndarray, dict]:
    """Crop one BGR frame and letterbox it without changing aspect ratio."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("person crop rendering requires OpenCV") from exc
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    height, width = frame.shape[:2]
    left, top, right, bottom = map(float, crop_xyxy)
    x1 = max(0, min(width - 1, int(np.floor(left))))
    y1 = max(0, min(height - 1, int(np.floor(top))))
    x2 = max(x1 + 1, min(width, int(np.ceil(right))))
    y2 = max(y1 + 1, min(height, int(np.ceil(bottom))))
    crop = frame[y1:y2, x1:x2]
    scale = min(output_size / crop.shape[1], output_size / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    canvas = np.full(
        (output_size, output_size, 3), padding_value, dtype=np.uint8
    )
    offset_x = (output_size - resized_width) // 2
    offset_y = (output_size - resized_height) // 2
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized
    return canvas, {
        "source_crop_xyxy": [x1, y1, x2, y2],
        "scale": float(scale),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
    }
