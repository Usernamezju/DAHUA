"""Sliding-window inference and visual evaluation for Set-aware Group STN."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from dahua_cup.pipeline.video_encoding import browser_video_args
from dahua_cup.stn.teacher_labels import validate_teacher_record


IMAGENET_MEAN = np.asarray(
    (0.485, 0.456, 0.406), dtype=np.float32
).reshape(1, 1, 3)
IMAGENET_STD = np.asarray(
    (0.229, 0.224, 0.225), dtype=np.float32
).reshape(1, 1, 3)
QUERY_COLORS = ((62, 220, 156), (239, 132, 255))
GROUP_COLOR = (35, 220, 255)
TEACHER_COLOR = (255, 184, 76)


def load_teacher_records(path: str | Path) -> list[dict]:
    """Load and validate YOLO track records without importing PyTorch."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"STN teacher JSONL not found: {source}")
    records = []
    identifiers = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                validate_teacher_record(record)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid STN teacher record at line {line_number}: {exc}"
                ) from exc
            sample_id = str(record["sample_id"])
            if sample_id in identifiers:
                raise ValueError(f"duplicate STN sample_id: {sample_id}")
            identifiers.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError("STN teacher JSONL contains no records")
    return records


def checkpoint_fingerprint(path: str | Path) -> str:
    """Return a cheap cache key tied to checkpoint path, size and mtime."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    value = f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def teacher_record_fingerprint(record: dict) -> str:
    """Return a stable cache key for one complete teacher record."""
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _clip_box(box: Sequence[float]) -> np.ndarray:
    value = np.asarray(box, dtype=np.float32).copy()
    value[[0, 2]] = np.clip(value[[0, 2]], 0.0, 1.0)
    value[[1, 3]] = np.clip(value[[1, 3]], 0.0, 1.0)
    if value[2] <= value[0]:
        value[0] = min(float(value[0]), 1.0 - 1e-5)
        value[2] = value[0] + 1e-5
    if value[3] <= value[1]:
        value[1] = min(float(value[1]), 1.0 - 1e-5)
        value[3] = value[1] + 1e-5
    return value


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """IoU for two normalized xyxy boxes."""
    first_value = _clip_box(first)
    second_value = _clip_box(second)
    left = max(float(first_value[0]), float(second_value[0]))
    top = max(float(first_value[1]), float(second_value[1]))
    right = min(float(first_value[2]), float(second_value[2]))
    bottom = min(float(first_value[3]), float(second_value[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = float(
        (first_value[2] - first_value[0])
        * (first_value[3] - first_value[1])
    )
    second_area = float(
        (second_value[2] - second_value[0])
        * (second_value[3] - second_value[1])
    )
    return intersection / max(first_area + second_area - intersection, 1e-8)


def _square_group_box(
    boxes: Sequence[Sequence[float]],
    *,
    context_factor: float,
    minimum_fraction: float,
) -> np.ndarray:
    if not boxes:
        return np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
    values = np.asarray(boxes, dtype=np.float32)
    merged = np.asarray(
        (
            values[:, 0].min(),
            values[:, 1].min(),
            values[:, 2].max(),
            values[:, 3].max(),
        ),
        dtype=np.float32,
    )
    center_x = float((merged[0] + merged[2]) / 2)
    center_y = float((merged[1] + merged[3]) / 2)
    side = min(
        1.0,
        max(
            minimum_fraction,
            float(max(merged[2] - merged[0], merged[3] - merged[1]))
            * context_factor,
        ),
    )
    half = side / 2
    center_x = min(max(center_x, half), 1.0 - half)
    center_y = min(max(center_y, half), 1.0 - half)
    return np.asarray(
        (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        ),
        dtype=np.float32,
    )


def letterbox_geometry(
    width: int, height: int, image_size: int
) -> dict[str, float]:
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return {
        "scale": float(scale),
        "left": float((image_size - resized_width) // 2),
        "top": float((image_size - resized_height) // 2),
        "size": float(image_size),
        "source_width": float(width),
        "source_height": float(height),
        "resized_width": float(resized_width),
        "resized_height": float(resized_height),
    }


def original_to_letterbox(
    box: Sequence[float], geometry: dict[str, float]
) -> np.ndarray:
    """Map an original normalized box to normalized square input space."""
    width = geometry["source_width"]
    height = geometry["source_height"]
    scale = geometry["scale"]
    size = geometry["size"]
    left = geometry["left"]
    top = geometry["top"]
    value = np.asarray(box, dtype=np.float32)
    return _clip_box(
        (
            (value[0] * width * scale + left) / size,
            (value[1] * height * scale + top) / size,
            (value[2] * width * scale + left) / size,
            (value[3] * height * scale + top) / size,
        )
    )


def letterbox_to_original(
    box: Sequence[float], geometry: dict[str, float]
) -> np.ndarray:
    """Map a normalized square-input box back to original video space."""
    value = np.asarray(box, dtype=np.float32)
    width = geometry["source_width"]
    height = geometry["source_height"]
    scale = geometry["scale"]
    size = geometry["size"]
    left = geometry["left"]
    top = geometry["top"]
    return _clip_box(
        (
            (value[0] * size - left) / (width * scale),
            (value[1] * size - top) / (height * scale),
            (value[2] * size - left) / (width * scale),
            (value[3] * size - top) / (height * scale),
        )
    )


def _contains(group: Sequence[float], person: Sequence[float]) -> bool:
    group_value = np.asarray(group)
    person_value = np.asarray(person)
    tolerance = 0.01
    return bool(
        group_value[0] <= person_value[0] + tolerance
        and group_value[1] <= person_value[1] + tolerance
        and group_value[2] >= person_value[2] - tolerance
        and group_value[3] >= person_value[3] - tolerance
    )


def evaluate_predictions(
    group_boxes: np.ndarray,
    presence_probabilities: np.ndarray,
    teacher_frames: Sequence[dict],
    geometry: dict[str, float],
    *,
    context_factor: float,
    minimum_fraction: float,
    presence_threshold: float,
) -> tuple[dict, list[dict]]:
    """Compare per-frame STN outputs with YOLO teacher tracks."""
    annotations = {
        int(frame["frame_index"]): frame for frame in teacher_frames
    }
    frame_details = []
    ious = []
    contained = []
    presence_correct = []
    zoom_factors = []
    annotated_frames = 0
    for frame_index in range(len(group_boxes)):
        frame = annotations.get(frame_index, {"persons": []})
        people = list(frame.get("persons") or [])
        teacher_boxes = [
            original_to_letterbox(person["bbox_xyxy"], geometry)
            for person in people
        ]
        prediction = _clip_box(group_boxes[frame_index])
        teacher_group = _square_group_box(
            teacher_boxes,
            context_factor=context_factor,
            minimum_fraction=minimum_fraction,
        )
        predicted_count = int(
            (presence_probabilities[frame_index] >= presence_threshold).sum()
        )
        teacher_count = len(people)
        presence_correct.append(float(predicted_count == teacher_count))
        side = max(
            float(prediction[2] - prediction[0]),
            float(prediction[3] - prediction[1]),
            1e-5,
        )
        zoom_factors.append(1.0 / side)
        frame_iou = None
        frame_containment = None
        if people:
            annotated_frames += 1
            frame_iou = box_iou(prediction, teacher_group)
            frame_containment = float(
                all(_contains(prediction, person) for person in teacher_boxes)
            )
            ious.append(frame_iou)
            contained.append(frame_containment)
        frame_details.append(
            {
                "frame_index": frame_index,
                "teacher_people": teacher_count,
                "predicted_people": predicted_count,
                "group_iou": frame_iou,
                "contained": frame_containment,
                "zoom_factor": zoom_factors[-1],
            }
        )

    center_size = np.stack(
        (
            (group_boxes[:, 0] + group_boxes[:, 2]) / 2,
            (group_boxes[:, 1] + group_boxes[:, 3]) / 2,
            group_boxes[:, 2] - group_boxes[:, 0],
            group_boxes[:, 3] - group_boxes[:, 1],
        ),
        axis=1,
    )
    jitter = (
        float(np.abs(np.diff(center_size, axis=0)).mean())
        if len(center_size) > 1
        else 0.0
    )
    worst = sorted(
        (
            detail
            for detail in frame_details
            if detail["group_iou"] is not None
        ),
        key=lambda item: item["group_iou"],
    )[:12]
    metrics = {
        "frames": int(len(group_boxes)),
        "annotated_frames": annotated_frames,
        "mean_group_iou": float(np.mean(ious)) if ious else None,
        "median_group_iou": float(np.median(ious)) if ious else None,
        "containment_recall": (
            float(np.mean(contained)) if contained else None
        ),
        "presence_accuracy": (
            float(np.mean(presence_correct)) if presence_correct else None
        ),
        "temporal_jitter": jitter,
        "mean_zoom_factor": float(np.mean(zoom_factors)),
        "maximum_zoom_factor": float(np.max(zoom_factors)),
        "worst_frames": worst,
    }
    return metrics, frame_details


def _interpolate_samples(
    values: np.ndarray,
    sample_indices: Sequence[int],
    frame_count: int,
) -> np.ndarray:
    if len(values) != len(sample_indices):
        raise ValueError("sample values and indices must have equal lengths")
    target = np.arange(frame_count, dtype=np.float32)
    source = np.asarray(sample_indices, dtype=np.float32)
    flat = values.reshape(len(values), -1)
    output = np.empty((frame_count, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target, source, flat[:, column])
    return output.reshape((frame_count,) + values.shape[1:])


def _window_starts(length: int, clip_length: int, step: int) -> list[int]:
    if length <= clip_length:
        return [0]
    starts = list(range(0, length - clip_length + 1, step))
    final = length - clip_length
    if starts[-1] != final:
        starts.append(final)
    return starts


def _draw_dashed_rectangle(
    frame,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
) -> None:
    import cv2

    left, top = first
    right, bottom = second
    for start in range(left, right, dash * 2):
        cv2.line(
            frame, (start, top), (min(start + dash, right), top),
            color, thickness, cv2.LINE_AA,
        )
        cv2.line(
            frame, (start, bottom), (min(start + dash, right), bottom),
            color, thickness, cv2.LINE_AA,
        )
    for start in range(top, bottom, dash * 2):
        cv2.line(
            frame, (left, start), (left, min(start + dash, bottom)),
            color, thickness, cv2.LINE_AA,
        )
        cv2.line(
            frame, (right, start), (right, min(start + dash, bottom)),
            color, thickness, cv2.LINE_AA,
        )


def _pixel_box(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    value = _clip_box(box)
    return (
        int(round(value[0] * (width - 1))),
        int(round(value[1] * (height - 1))),
        int(round(value[2] * (width - 1))),
        int(round(value[3] * (height - 1))),
    )


def _ffmpeg_writer(
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> subprocess.Popen:
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "-", "-an",
    ]
    command.extend(
        browser_video_args(codec, "veryfast", quality=23, bitrate="3M")
    )
    command.append(str(output))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


@dataclass(frozen=True)
class STNInferenceOptions:
    device: str = "cuda:0"
    batch_size: int = 4
    window_step: int = 8
    presence_threshold: float = 0.5
    overlay_width: int = 960
    crop_size: int = 640
    ffmpeg: str = "ffmpeg"
    codec: str = "libopenh264"
    amp: bool = True


class STNVideoRunner:
    """Load one checkpoint lazily and render auditable STN predictions."""

    def __init__(
        self,
        checkpoint: str | Path,
        output_root: str | Path,
        options: STNInferenceOptions,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.options = options
        self._model = None
        self._torch = None
        self._config = None

    def _load_model(self):
        if self._model is not None:
            return self._model, self._torch, self._config
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"STN checkpoint not found: {self.checkpoint}"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("STN inference requires PyTorch") from exc
        from dahua_cup.stn.model import SetAwareGroupSTN

        device = torch.device(self.options.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device is unavailable: {device}")
        checkpoint = torch.load(str(self.checkpoint), map_location=device)
        config = checkpoint.get("config")
        if not isinstance(config, dict) or not isinstance(
            config.get("model"), dict
        ):
            raise ValueError("STN checkpoint does not contain model config")
        model = SetAwareGroupSTN(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        self._model = model
        self._torch = torch
        self._config = config
        return model, torch, config

    @staticmethod
    def _preprocess_frame(frame, image_size: int):
        import cv2

        height, width = frame.shape[:2]
        geometry = letterbox_geometry(width, height, image_size)
        resized = cv2.resize(
            frame,
            (
                int(geometry["resized_width"]),
                int(geometry["resized_height"]),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
        left = int(geometry["left"])
        top = int(geometry["top"])
        canvas[
            top:top + resized.shape[0], left:left + resized.shape[1]
        ] = resized
        rgb = canvas[..., ::-1].astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)), geometry

    def _decode_samples(
        self, video: Path, image_size: int, frame_stride: int
    ) -> tuple[np.ndarray, list[int], dict]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("STN visualization requires OpenCV") from exc
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open STN video: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError(f"invalid video metadata: {video}")
        samples = []
        indices = []
        frame_index = 0
        geometry = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                sample, geometry = self._preprocess_frame(frame, image_size)
                samples.append(sample)
                indices.append(frame_index)
            frame_index += 1
        capture.release()
        if not samples or geometry is None:
            raise RuntimeError(f"video contains no decodable frames: {video}")
        return np.stack(samples), indices, {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_index,
            "geometry": geometry,
        }

    def _predict_samples(
        self, samples: np.ndarray, clip_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model, torch, _config = self._load_model()
        length = len(samples)
        starts = _window_starts(
            length, clip_length, self.options.window_step
        )
        box_sum = np.zeros((length, 2, 4), dtype=np.float64)
        group_sum = np.zeros((length, 4), dtype=np.float64)
        presence_sum = np.zeros((length, 2), dtype=np.float64)
        counts = np.zeros(length, dtype=np.float64)
        device = torch.device(self.options.device)
        amp_enabled = self.options.amp and device.type == "cuda"
        for batch_start in range(0, len(starts), self.options.batch_size):
            batch_starts = starts[
                batch_start:batch_start + self.options.batch_size
            ]
            windows = []
            positions = []
            for start in batch_starts:
                selected = [
                    min(start + offset, length - 1)
                    for offset in range(clip_length)
                ]
                windows.append(samples[selected])
                positions.append(selected)
            tensor = torch.from_numpy(np.stack(windows)).permute(
                0, 2, 1, 3, 4
            ).to(device=device, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(tensor, apply_transform=False)
                boxes = outputs["boxes_xyxy"].float().cpu().numpy()
                groups = (
                    outputs["group_boxes_xyxy"].float().cpu().numpy()
                )
                probabilities = torch.sigmoid(
                    outputs["presence_logits"]
                ).float().cpu().numpy()
            for window_index, selected in enumerate(positions):
                for time_index, sample_index in enumerate(selected):
                    box_sum[sample_index] += boxes[
                        window_index, time_index
                    ]
                    group_sum[sample_index] += groups[
                        window_index, time_index
                    ]
                    presence_sum[sample_index] += probabilities[
                        window_index, time_index
                    ]
                    counts[sample_index] += 1
        counts = np.maximum(counts, 1.0)
        return (
            (box_sum / counts[:, None, None]).astype(np.float32),
            (group_sum / counts[:, None]).astype(np.float32),
            (presence_sum / counts[:, None]).astype(np.float32),
        )

    def _render(
        self,
        video: Path,
        overlay_output: Path,
        crop_output: Path,
        group_boxes: np.ndarray,
        query_boxes: np.ndarray,
        presence: np.ndarray,
        teacher_frames: Sequence[dict],
        frame_details: Sequence[dict],
        metadata: dict,
    ) -> None:
        import cv2

        width = int(metadata["width"])
        height = int(metadata["height"])
        fps = float(metadata["fps"])
        overlay_width = min(self.options.overlay_width, width)
        overlay_width = max(2, overlay_width - overlay_width % 2)
        overlay_height = int(round(height * overlay_width / width))
        overlay_height = max(2, overlay_height - overlay_height % 2)
        crop_size = max(2, self.options.crop_size)
        crop_size -= crop_size % 2
        overlay_temporary = overlay_output.with_name(
            overlay_output.stem + ".tmp.mp4"
        )
        crop_temporary = crop_output.with_name(
            crop_output.stem + ".tmp.mp4"
        )
        overlay_writer = _ffmpeg_writer(
            self.options.ffmpeg,
            overlay_temporary,
            overlay_width,
            overlay_height,
            fps,
            self.options.codec,
        )
        crop_writer = _ffmpeg_writer(
            self.options.ffmpeg,
            crop_temporary,
            crop_size,
            crop_size,
            fps,
            self.options.codec,
        )
        assert overlay_writer.stdin is not None
        assert crop_writer.stdin is not None
        annotations = {
            int(frame["frame_index"]): frame for frame in teacher_frames
        }
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot reopen STN video: {video}")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                overlay = cv2.resize(
                    frame, (overlay_width, overlay_height),
                    interpolation=cv2.INTER_AREA,
                )
                for query_index, color in enumerate(QUERY_COLORS):
                    probability = float(presence[frame_index, query_index])
                    if probability < self.options.presence_threshold:
                        continue
                    left, top, right, bottom = _pixel_box(
                        query_boxes[frame_index, query_index],
                        overlay_width,
                        overlay_height,
                    )
                    cv2.rectangle(
                        overlay, (left, top), (right, bottom),
                        color, 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        f"Q{query_index + 1} {probability:.2f}",
                        (left, max(18, top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                group = group_boxes[frame_index]
                left, top, right, bottom = _pixel_box(
                    group, overlay_width, overlay_height
                )
                cv2.rectangle(
                    overlay, (left, top), (right, bottom),
                    GROUP_COLOR, 3, cv2.LINE_AA,
                )
                teacher = annotations.get(frame_index, {"persons": []})
                for person in teacher.get("persons") or []:
                    teacher_box = _pixel_box(
                        person["bbox_xyxy"], overlay_width, overlay_height
                    )
                    _draw_dashed_rectangle(
                        overlay,
                        teacher_box[:2],
                        teacher_box[2:],
                        TEACHER_COLOR,
                    )
                detail = frame_details[frame_index]
                text = (
                    f"Frame {frame_index + 1}/{metadata['frame_count']}  "
                    f"people {detail['predicted_people']}/"
                    f"{detail['teacher_people']}  "
                    f"zoom {detail['zoom_factor']:.2f}x"
                )
                if detail["group_iou"] is not None:
                    text += f"  IoU {detail['group_iou']:.3f}"
                cv2.rectangle(
                    overlay, (0, 0), (overlay_width, 38),
                    (12, 17, 27), -1,
                )
                cv2.putText(
                    overlay, text, (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (235, 241, 249), 2, cv2.LINE_AA,
                )
                overlay_writer.stdin.write(overlay.tobytes())

                crop_left, crop_top, crop_right, crop_bottom = _pixel_box(
                    group, width, height
                )
                if crop_right <= crop_left or crop_bottom <= crop_top:
                    crop = frame
                else:
                    crop = frame[
                        crop_top:crop_bottom + 1,
                        crop_left:crop_right + 1,
                    ]
                crop = cv2.resize(
                    crop, (crop_size, crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                cv2.rectangle(
                    crop, (0, 0), (crop_size, 34), (12, 17, 27), -1
                )
                cv2.putText(
                    crop,
                    f"STN crop  {detail['zoom_factor']:.2f}x",
                    (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (235, 241, 249),
                    2,
                    cv2.LINE_AA,
                )
                crop_writer.stdin.write(crop.tobytes())
                frame_index += 1
        finally:
            capture.release()
            overlay_writer.stdin.close()
            crop_writer.stdin.close()
        overlay_code = overlay_writer.wait()
        crop_code = crop_writer.wait()
        if overlay_code or crop_code:
            raise RuntimeError(
                "FFmpeg STN renderer failed: "
                f"overlay={overlay_code}, crop={crop_code}"
            )
        overlay_temporary.replace(overlay_output)
        crop_temporary.replace(crop_output)

    def run(self, record: dict, *, force: bool = False) -> dict:
        validate_teacher_record(record)
        video = Path(record["video_path"]).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"STN source video not found: {video}")
        artifact_root = self.output_root / _safe_name(record["sample_id"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        result_path = artifact_root / "result.json"
        fingerprint = checkpoint_fingerprint(self.checkpoint)
        record_fingerprint = teacher_record_fingerprint(record)
        if result_path.is_file() and not force:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                cached.get("checkpoint_fingerprint") == fingerprint
                and cached.get("teacher_record_fingerprint")
                == record_fingerprint
                and (artifact_root / "overlay.mp4").is_file()
                and (artifact_root / "crop.mp4").is_file()
            ):
                return cached

        model, _torch, config = self._load_model()
        del model
        data_config = config["data"]
        model_config = config["model"]
        image_size = int(data_config["image_size"])
        frame_stride = int(data_config["frame_stride"])
        clip_length = int(data_config["clip_length"])
        samples, sample_indices, metadata = self._decode_samples(
            video, image_size, frame_stride
        )
        sample_query, sample_group, sample_presence = self._predict_samples(
            samples, clip_length
        )
        frame_count = int(metadata["frame_count"])
        query_letterbox = _interpolate_samples(
            sample_query, sample_indices, frame_count
        )
        group_letterbox = _interpolate_samples(
            sample_group, sample_indices, frame_count
        )
        presence = _interpolate_samples(
            sample_presence, sample_indices, frame_count
        )
        geometry = metadata["geometry"]
        query_original = np.stack(
            [
                [
                    letterbox_to_original(box, geometry)
                    for box in frame_boxes
                ]
                for frame_boxes in query_letterbox
            ]
        )
        group_original = np.stack(
            [
                letterbox_to_original(box, geometry)
                for box in group_letterbox
            ]
        )
        metrics, frame_details = evaluate_predictions(
            group_letterbox,
            presence,
            record["frames"],
            geometry,
            context_factor=float(model_config["context_factor"]),
            minimum_fraction=float(
                model_config["minimum_crop_fraction"]
            ),
            presence_threshold=self.options.presence_threshold,
        )
        overlay_path = artifact_root / "overlay.mp4"
        crop_path = artifact_root / "crop.mp4"
        self._render(
            video,
            overlay_path,
            crop_path,
            group_original,
            query_original,
            presence,
            record["frames"],
            frame_details,
            metadata,
        )
        result = {
            "schema_version": "stn_visualization.v1",
            "sample_id": str(record["sample_id"]),
            "source_dataset": str(
                record.get("source_dataset") or "unknown"
            ),
            "video_path": str(video),
            "checkpoint": str(self.checkpoint),
            "checkpoint_fingerprint": fingerprint,
            "teacher_record_fingerprint": record_fingerprint,
            "device": self.options.device,
            "metrics": metrics,
            "capacity": {
                "maximum_people": 2,
                "crowd_frame_ratio": float(
                    record.get("crowd_frame_ratio") or 0.0
                ),
                "capacity_exceeded": float(
                    record.get("crowd_frame_ratio") or 0.0
                ) > 0.0,
                "metrics_cover_selected_top2_only": True,
            },
            "artifacts": {
                "overlay": str(overlay_path),
                "crop": str(crop_path),
            },
            "inference": {
                "clip_length": clip_length,
                "frame_stride": frame_stride,
                "window_step": self.options.window_step,
                "sampled_frames": len(sample_indices),
                "presence_threshold": self.options.presence_threshold,
            },
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(result_path)
        return result


def _safe_name(value: object) -> str:
    text = str(value)
    clean = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in text
    ).strip("._")
    if not clean:
        raise ValueError("sample_id cannot be converted to an artifact name")
    return clean[:180]


def summarize_record(record: dict) -> dict:
    """Return lightweight metadata for the standalone sample browser."""
    people_per_frame = [
        len(frame.get("persons") or []) for frame in record["frames"]
    ]
    confidences = [
        float(person["confidence"])
        for frame in record["frames"]
        for person in frame.get("persons") or []
    ]
    return {
        "sample_id": str(record["sample_id"]),
        "source_dataset": str(record.get("source_dataset") or "unknown"),
        "video_path": str(record["video_path"]),
        "selected_people": len(record.get("selected_track_ids") or []),
        "frames": len(record["frames"]),
        "annotated_frame_ratio": float(
            np.mean([value > 0 for value in people_per_frame])
        ),
        "mean_teacher_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "crowd_frame_ratio": float(
            record.get("crowd_frame_ratio") or 0.0
        ),
        "capacity_exceeded": float(
            record.get("crowd_frame_ratio") or 0.0
        ) > 0.0,
        "video_exists": Path(record["video_path"]).expanduser().is_file(),
    }
