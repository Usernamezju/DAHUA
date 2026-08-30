"""Video dataset for frozen-YOLO response distillation."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dahua_cup.pipeline.common import read_jsonl
from dahua_cup.stn.teacher_labels import validate_teacher_record


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_teacher_records(path: str | Path) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        validate_teacher_record(row)
    return rows


def split_teacher_records(
    rows: Sequence[dict],
    *,
    validation_fraction: float = 0.15,
    seed: int = 20260729,
) -> tuple[list[dict], list[dict]]:
    """Stable sample-level split that never leaks clips across partitions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    train, validation = [], []
    group_destination = {}
    for row in rows:
        group = str(row.get("split_group") or row["sample_id"])
        key = f"{seed}:{group}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        is_validation = value / float(2**64) < validation_fraction
        group_destination.setdefault(group, is_validation)
        destination = validation if group_destination[group] else train
        destination.append(row)
    def move_one_group(source: list[dict], destination: list[dict]) -> None:
        group = str(
            source[-1].get("split_group") or source[-1]["sample_id"]
        )
        moving = [
            row
            for row in source
            if str(row.get("split_group") or row["sample_id"]) == group
        ]
        source[:] = [row for row in source if row not in moving]
        destination.extend(moving)

    if len(rows) > 1 and not validation:
        move_one_group(train, validation)
    if len(rows) > 1 and not train:
        move_one_group(validation, train)
    if not train or not validation:
        raise ValueError(
            "STN validation requires at least two independent split groups"
        )
    return train, validation


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("STN video loading requires opencv-python") from exc
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, {
        "scale": scale,
        "left": left,
        "top": top,
        "source_width": width,
        "source_height": height,
        "size": size,
    }


def _letterbox_box(box: Sequence[float], geometry: dict) -> np.ndarray:
    width = geometry["source_width"]
    height = geometry["source_height"]
    scale = geometry["scale"]
    size = geometry["size"]
    left = geometry["left"]
    top = geometry["top"]
    values = np.asarray(box, dtype=np.float32).copy()
    values[[0, 2]] = (values[[0, 2]] * width * scale + left) / size
    values[[1, 3]] = (values[[1, 3]] * height * scale + top) / size
    return np.clip(values, 0.0, 1.0)


def _scale_canvas(
    frames: list[np.ndarray],
    boxes: np.ndarray,
    factor: float,
    rng: random.Random,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Shrink the entire clip to synthesize 1x/2x/4x person scales."""
    if factor == 1.0:
        return frames, boxes
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("STN scale augmentation requires OpenCV") from exc
    if not 0 < factor < 1:
        raise ValueError("synthetic scale must be in (0,1]")
    size = frames[0].shape[0]
    scaled_size = max(1, int(round(size * factor)))
    maximum_offset = size - scaled_size
    offset_x = rng.randint(0, maximum_offset)
    offset_y = rng.randint(0, maximum_offset)
    output = []
    for frame in frames:
        resized = cv2.resize(
            frame,
            (scaled_size, scaled_size),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full_like(frame, 114)
        canvas[
            offset_y:offset_y + scaled_size,
            offset_x:offset_x + scaled_size,
        ] = resized
        output.append(canvas)
    transformed = boxes.copy()
    transformed[..., [0, 2]] = (
        offset_x / size + factor * transformed[..., [0, 2]]
    )
    transformed[..., [1, 3]] = (
        offset_y / size + factor * transformed[..., [1, 3]]
    )
    return output, np.clip(transformed, 0.0, 1.0)


class YOLOTrackDistillationDataset(Dataset):
    """Decode clips and align YOLO track boxes to two set targets."""

    def __init__(
        self,
        rows: Sequence[dict],
        *,
        clip_length: int = 16,
        frame_stride: int = 2,
        image_size: int = 128,
        training: bool = True,
        scale_factors: Sequence[float] = (1.0, 0.5, 0.25),
        horizontal_flip_probability: float = 0.5,
        seed: int = 20260729,
    ) -> None:
        if not rows:
            raise ValueError("STN dataset contains no records")
        if clip_length < 2 or frame_stride < 1 or image_size < 32:
            raise ValueError("invalid clip length, stride, or image size")
        factors = tuple(float(value) for value in scale_factors)
        if not factors or any(not 0 < value <= 1 for value in factors):
            raise ValueError("scale factors must be in (0,1]")
        self.rows = list(rows)
        for row in self.rows:
            validate_teacher_record(row)
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.training = training
        self.scale_factors = factors
        self.horizontal_flip_probability = (
            horizontal_flip_probability if training else 0.0
        )
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        if self.training:
            return random.Random(
                self.seed
                + index * 1_000_003
                + random.randrange(0, 2**30)
            )
        return random.Random(self.seed + index)

    def _indices(self, row: dict, rng: random.Random) -> list[int]:
        total = int(row["total_frames"])
        span = 1 + (self.clip_length - 1) * self.frame_stride
        if total >= span:
            start = rng.randint(0, total - span) if self.training else (
                total - span
            ) // 2
            return [
                start + offset * self.frame_stride
                for offset in range(self.clip_length)
            ]
        return np.linspace(
            0, max(0, total - 1), self.clip_length
        ).round().astype(int).tolist()

    @staticmethod
    def _decode_frames(path: Path, indices: Sequence[int]) -> list[np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("STN video loading requires OpenCV") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open STN training video: {path}")
        required = set(map(int, indices))
        decoded = {}
        frame_index = 0
        try:
            while required:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in required:
                    decoded[frame_index] = frame
                    required.remove(frame_index)
                frame_index += 1
        finally:
            capture.release()
        if required:
            raise RuntimeError(
                f"video ended before frames {sorted(required)}: {path}"
            )
        return [decoded[int(value)] for value in indices]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = Path(row["video_path"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"STN training video not found: {path}")
        rng = self._rng(index)
        indices = self._indices(row, rng)
        decoded = self._decode_frames(path, indices)
        letterboxed = []
        geometries = []
        for frame in decoded:
            transformed, geometry = _letterbox(frame, self.image_size)
            letterboxed.append(transformed)
            geometries.append(geometry)

        annotations = {
            int(frame["frame_index"]): frame
            for frame in row["frames"]
        }
        track_ids = list(map(int, row["selected_track_ids"]))[:2]
        track_to_slot = {
            track_id: slot for slot, track_id in enumerate(track_ids)
        }
        boxes = np.zeros((self.clip_length, 2, 4), dtype=np.float32)
        presence = np.zeros((self.clip_length, 2), dtype=bool)
        confidence = np.zeros((self.clip_length, 2), dtype=np.float32)
        for time_index, (frame_index, geometry) in enumerate(
            zip(indices, geometries)
        ):
            for person in annotations.get(
                int(frame_index), {"persons": []}
            )["persons"]:
                slot = track_to_slot.get(int(person["track_id"]))
                if slot is None:
                    continue
                boxes[time_index, slot] = _letterbox_box(
                    person["bbox_xyxy"], geometry
                )
                presence[time_index, slot] = True
                confidence[time_index, slot] = float(person["confidence"])

        factor = rng.choice(self.scale_factors) if self.training else 1.0
        letterboxed, boxes = _scale_canvas(
            letterboxed, boxes, factor, rng
        )
        if (
            self.training
            and rng.random() < self.horizontal_flip_probability
        ):
            letterboxed = [
                np.ascontiguousarray(frame[:, ::-1])
                for frame in letterboxed
            ]
            old_left = boxes[..., 0].copy()
            boxes[..., 0] = 1.0 - boxes[..., 2]
            boxes[..., 2] = 1.0 - old_left

        rgb = np.stack(
            [frame[..., ::-1] for frame in letterboxed],
            axis=0,
        ).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        video = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(3, 0, 1, 2))
        )
        return {
            "sample_id": str(row["sample_id"]),
            "video": video,
            "boxes_xyxy": torch.from_numpy(boxes),
            "presence": torch.from_numpy(presence),
            "confidence": torch.from_numpy(confidence),
            "scale_factor": torch.tensor(factor, dtype=torch.float32),
            "frame_indices": torch.tensor(indices, dtype=torch.long),
        }
