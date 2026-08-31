"""Lazy, local-only Qwen3-VL teacher backend.

Heavy dependencies and model files are touched only when ``load`` or ``predict``
is called, so the rest of the pipeline remains testable without model weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dahua_cup.semantic_teacher.prompts.prompt_builder import PROMPT_VERSION, build_teacher_prompt
from dahua_cup.semantic_teacher.schemas import (
    LABELS,
    TeacherOutput,
    normalize_distribution,
)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    content = fenced.group(1) if fenced else text
    candidate = content[content.find("{"): content.rfind("}") + 1]
    if not candidate or candidate[0] != "{":
        raise ValueError("teacher response does not contain a JSON object")
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid teacher JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("teacher JSON must be an object")
    return value


SCHEMA_VERSION_ALIASES = {"v1": "teacher_output.v1"}


def parse_teacher_output(
    value: Mapping[str, Any], allowed_labels: Sequence[str]
) -> tuple[TeacherOutput, dict[str, Any] | None]:
    """Validate a teacher response and canonicalize it.

    The prompt contract no longer asks the model for a label: the label is
    derived deterministically as the argmax of the distribution, so the two
    fields can never disagree.  Responses that still carry a label (older
    prompts, or a model that insists) are canonicalized the same way, with a
    conflicting raw label preserved in provenance for audit.

    The prompt permits a sparse distribution, which the schema normalizes
    over the full closed set.  A model can therefore report a confidence
    that was correct before normalization but differs afterwards.  Keep that
    raw, uncalibrated value in provenance while using the normalized
    selected-label probability everywhere the schema requires confidence.
    Other schema violations remain hard failures and are retried.
    """
    candidate = dict(value)
    repairs: dict[str, Any] = {}

    raw_schema = candidate.get("schema_version")
    if isinstance(raw_schema, str) and raw_schema in SCHEMA_VERSION_ALIASES:
        candidate["schema_version"] = SCHEMA_VERSION_ALIASES[raw_schema]
        repairs["schema_version_normalized_from"] = raw_schema

    distribution = normalize_distribution(
        candidate.get("distribution", {}), allowed_labels
    )
    derived_label = max(distribution, key=distribution.get)
    raw_label = candidate.get("label")
    if raw_label is not None and str(raw_label) != derived_label:
        repairs["raw_label"] = raw_label
    candidate["label"] = derived_label

    try:
        output = TeacherOutput.from_dict(candidate, allowed_labels=allowed_labels)
    except ValueError as exc:
        if str(exc) != "confidence must match the selected label probability":
            raise
        candidate["confidence"] = distribution.get(derived_label, -1.0)
        output = TeacherOutput.from_dict(candidate, allowed_labels=allowed_labels)
        try:
            repairs["raw_confidence"] = float(value.get("confidence"))
        except (TypeError, ValueError):
            pass
    return output, (repairs or None)


def local_video_reference(path: str | Path) -> str:
    """Return the absolute local path expected by Transformers video loaders."""
    video = Path(path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"teacher video not found: {video}")
    return str(video)


def decode_video_frames(
    path: str | Path, max_frames: int
) -> tuple[Any, dict[str, Any]]:
    """Decode uniformly sampled RGB frames without Transformers video I/O.

    Some minimal torchvision builds expose ``torchvision.io`` but omit
    ``read_video``.  Passing a path to ``apply_chat_template`` then fails
    before Qwen inference starts.  OpenCV is already a pipeline dependency,
    and Qwen3-VL processors natively accept uint8 NumPy videos.
    """
    if max_frames < 2:
        raise ValueError("Qwen video inference requires max_frames >= 2")
    video = local_video_reference(path)
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "teacher video decoding requires OpenCV and NumPy"
        ) from exc

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open teacher video: {video}")
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_frames <= 0:
        capture.release()
        raise RuntimeError(
            f"teacher video reports no decodable frames: {video}"
        )

    sample_count = min(max_frames, source_frames)
    if sample_count > 1 and sample_count % 2:
        sample_count -= 1
    target_indices = np.linspace(
        0, source_frames - 1, num=max(1, sample_count)
    ).round().astype(np.int64).tolist()
    target_set = set(target_indices)
    frames = []
    decoded_index = 0
    try:
        while decoded_index <= target_indices[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_index in target_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            decoded_index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"OpenCV decoded no teacher video frames: {video}")
    if len(frames) == 1:
        frames.append(frames[0].copy())
        target_indices.append(target_indices[0])
    if len(frames) % 2:
        frames.append(frames[-1].copy())
        target_indices.append(target_indices[-1])
    array = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    return array, {
        "decoder": "opencv",
        "source_frame_count": source_frames,
        "source_fps": source_fps,
        "sampled_frame_count": int(array.shape[0]),
        "sampled_frame_indices": target_indices[: int(array.shape[0])],
    }


def chat_template_kwargs(max_frames: int, has_video: bool) -> dict[str, Any]:
    """Build chat-template options for current multimodal processors."""
    options: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if has_video:
        options["processor_kwargs"] = {
            "num_frames": max_frames,
            "fps": None,
        }
    return options


@dataclass(frozen=True)
class Qwen3Config:
    model_dir: str
    revision: str | None = None
    dtype: str = "float16"
    device_map: str = "auto"
    attn_implementation: str | None = "sdpa"
    local_files_only: bool = True
    max_new_tokens: int = 512
    max_frames: int = 8
    retries: int = 2


class Qwen3Teacher:
    def __init__(self, config: Qwen3Config):
        self.config = config
        self.model = None
        self.processor = None

    def load(self) -> None:
        model_dir = Path(self.config.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Qwen3-VL model directory not found: {model_dir}")
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL inference requires torch and a Qwen3-VL capable transformers") from exc
        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.config.dtype}")
        common = {
            "revision": self.config.revision,
            "local_files_only": self.config.local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(str(model_dir), **common)
        model_kwargs = dict(common, torch_dtype=dtype, device_map=self.config.device_map)
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **model_kwargs)
        self.model.eval()

    def predict(
        self,
        sample_id: str,
        semantic_graph: Mapping[str, Any],
        student_distribution: Mapping[str, float] | None = None,
        images: Sequence[Any] | None = None,
        video_path: str | Path | None = None,
        allowed_labels: Sequence[str] = LABELS,
        task: str = "campus6",
    ) -> tuple[TeacherOutput, dict[str, Any]]:
        if self.model is None or self.processor is None:
            self.load()
        labels = tuple(str(label) for label in allowed_labels)
        prompt = build_teacher_prompt(
            sample_id,
            semantic_graph,
            student_distribution,
            allowed_labels=labels,
            task=task,
        )
        content = [{"type": "image", "image": image} for image in (images or [])]
        video_frames = None
        video_provenance = None
        if video_path is not None:
            video_frames, video_provenance = decode_video_frames(
                video_path, self.config.max_frames
            )
            content.append({
                "type": "video",
                "video": video_frames,
            })
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        template_kwargs = chat_template_kwargs(
            self.config.max_frames, video_path is not None
        )
        if self.config.retries < 0:
            raise ValueError("teacher retries must be non-negative")
        response = ""
        result = None
        repairs = None
        attempts = 0
        for attempts in range(1, self.config.retries + 2):
            if video_frames is None:
                inputs = self.processor.apply_chat_template(
                    messages, **template_kwargs
                )
            else:
                # Rendering the chat template without tokenization only emits
                # the video placeholder.  Supplying the already decoded array
                # directly to the processor bypasses torchcodec/torchvision.
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                processor_inputs: dict[str, Any] = {
                    "text": [prompt_text],
                    "videos": [video_frames],
                    "do_sample_frames": False,
                    "return_tensors": "pt",
                }
                if images:
                    processor_inputs["images"] = list(images)
                inputs = self.processor(**processor_inputs)
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.model.device)
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
            )
            input_length = inputs["input_ids"].shape[1]
            response = self.processor.batch_decode(
                generated[:, input_length:], skip_special_tokens=True
            )[0]
            try:
                result, repairs = parse_teacher_output(
                    extract_json_object(response), allowed_labels=labels
                )
                break
            except (TypeError, ValueError) as exc:
                if attempts > self.config.retries:
                    raise
                messages.extend(
                    (
                        {"role": "assistant", "content": [{"type": "text", "text": response}]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "The previous JSON failed validation: "
                                        f"{exc}. Return only a corrected JSON object."
                                    ),
                                }
                            ],
                        },
                    )
                )
        if result is None:
            raise RuntimeError("teacher inference produced no validated result")
        provenance = {
            "prompt_version": PROMPT_VERSION,
            "teacher_model_version": self.config.revision or Path(self.config.model_dir).name,
            "teacher_model_dir": str(Path(self.config.model_dir).resolve()),
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "attn_implementation": self.config.attn_implementation,
            "raw_response_hash": hashlib.sha256(response.encode()).hexdigest(),
            "attempts": attempts,
        }
        provenance["label_method"] = "distribution_argmax"
        if repairs:
            if "raw_confidence" in repairs:
                provenance["raw_confidence"] = repairs["raw_confidence"]
                provenance["confidence_method"] = (
                    "normalized_teacher_distribution; "
                    "raw_confidence_preserved_uncalibrated"
                )
            else:
                provenance["confidence_method"] = "teacher_self_report_uncalibrated"
            if "raw_label" in repairs:
                provenance["raw_label"] = repairs["raw_label"]
            if "schema_version_normalized_from" in repairs:
                provenance["schema_version_normalized_from"] = repairs[
                    "schema_version_normalized_from"
                ]
        else:
            provenance["confidence_method"] = "teacher_self_report_uncalibrated"
        if video_provenance is not None:
            provenance["video_decode"] = video_provenance
        return result, provenance
