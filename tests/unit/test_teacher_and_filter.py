import json

import numpy as np
import pytest

from dahua_cup.semantic_teacher.api.qwen3_backend import (
    Qwen3Config,
    chat_template_kwargs,
    decode_video_frames,
    extract_json_object,
    local_video_reference,
    parse_teacher_output,
)
from dahua_cup.semantic_teacher.api.qwen_api import QwenTeacher
from dahua_cup.semantic_teacher.prompts.prompt_builder import build_teacher_prompt
from dahua_cup.semantic_teacher.pseudo_label.filter_label import (
    PseudoLabelFilter,
    distribution_similarity,
)
from dahua_cup.semantic_teacher.schemas import LABELS, TeacherOutput
from dahua_cup.pipeline.qwen_teacher_worker import (
    build_parser,
    load_student_hint,
    summarize_pose_feature,
)


def distribution(label, probability=0.9):
    rest = (1 - probability) / (len(LABELS) - 1)
    return {item: probability if item == label else rest for item in LABELS}


def teacher(label="normal_walk", probability=0.9, review=False):
    return TeacherOutput("sample", label, distribution(label, probability), probability,
                         reason="measured evidence", needs_review=review)


def test_teacher_schema_rejects_argmax_mismatch():
    value = teacher().to_dict()
    value["label"] = "normal_run"
    with pytest.raises(ValueError, match="argmax"):
        TeacherOutput.from_dict(value)


def test_prompt_contains_structured_input_and_guardrails():
    prompt = build_teacher_prompt("sample", {"segments": [{"id": "s0"}]}, distribution("normal_walk"))
    assert "campus6-v1.0" in prompt
    assert "Do not invent" in prompt
    assert '"sample_id": "sample"' in prompt


def test_teacher_supports_task_specific_label_space():
    labels = ("walk", "run", "jump")
    value = {
        "schema_version": "teacher_output.v1",
        "sample_id": "sample",
        "label": "run",
        "distribution": [{"label": "run", "probability": 0.8}, {"label": "walk", "probability": 0.2}],
        "confidence": 0.8,
        "evidence": [],
        "counter_evidence": [],
        "reason": "visible motion",
        "needs_review": False,
    }
    output = TeacherOutput.from_dict(value, allowed_labels=labels)
    assert tuple(output.distribution) == labels
    assert output.label == "run"
    prompt = build_teacher_prompt(
        "sample", {"segments": [{"id": "s0"}]}, {"run": 0.7},
        allowed_labels=labels, task="three_class",
    )
    assert "Allowed labels (3)" in prompt
    assert "three_class" in prompt


def test_sparse_teacher_distribution_canonicalizes_confidence():
    value = {
        "schema_version": "teacher_output.v1",
        "sample_id": "sample",
        "label": "run",
        "distribution": [{"label": "run", "probability": 0.8}, {"label": "walk", "probability": 0.1}],
        "confidence": 0.8,
        "evidence": [],
        "counter_evidence": [],
        "reason": "visible motion",
        "needs_review": False,
    }
    output, raw_confidence = parse_teacher_output(value, ("walk", "run", "jump"))
    assert raw_confidence == pytest.approx(0.8)
    assert output.confidence == pytest.approx(0.8 / 0.9)


def test_qwen_teacher_defaults_to_native_sdpa(monkeypatch):
    monkeypatch.delenv("DAHUA_QWEN_ATTN_IMPLEMENTATION", raising=False)
    args = build_parser().parse_args([
        "--sample-id", "sample",
        "--feature", "feature.npz",
        "--pose-video", "pose.mp4",
        "--output", "teacher.json",
    ])
    assert args.attn_implementation == "sdpa"
    assert Qwen3Config(model_dir="model").attn_implementation == "sdpa"


def test_qwen_local_video_uses_path_and_nested_processor_kwargs(tmp_path):
    video = tmp_path / "pose video.mp4"
    video.write_bytes(b"video")
    reference = local_video_reference(video)
    assert reference == str(video.resolve())
    assert not reference.startswith("file:")
    options = chat_template_kwargs(max_frames=8, has_video=True)
    assert options["processor_kwargs"] == {"num_frames": 8, "fps": None}


def test_qwen_video_is_uniformly_predecoded_with_opencv(
    tmp_path, monkeypatch
):
    import sys
    import types

    video = tmp_path / "pose.mp4"
    video.write_bytes(b"video")
    source = [
        np.full((2, 3, 3), index, dtype=np.uint8)
        for index in range(10)
    ]

    class Capture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, key):
            return 10 if key == 1 else 25.0

        def read(self):
            if self.index >= len(source):
                return False, None
            frame = source[self.index]
            self.index += 1
            return True, frame

        def release(self):
            return None

    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FPS=2,
        COLOR_BGR2RGB=3,
        VideoCapture=Capture,
        cvtColor=lambda frame, _code: frame[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    frames, provenance = decode_video_frames(video, max_frames=8)

    assert frames.shape == (8, 2, 3, 3)
    assert frames.dtype == np.uint8
    assert provenance["decoder"] == "opencv"
    assert provenance["sampled_frame_indices"] == [0, 1, 3, 4, 5, 6, 8, 9]


def test_teacher_worker_builds_summary_and_reads_student_hint(tmp_path):
    feature = tmp_path / "pose.npz"
    valid = np.ones((2, 4, 25), dtype=bool)
    points = np.zeros((2, 4, 25, 2), dtype="float32")
    points[0, :, :, 0] = np.arange(4)[:, None]
    np.savez_compressed(
        feature, valid_mask=valid, image_keypoint=points, fps=2.0, width=10, height=10,
    )
    graph = summarize_pose_feature("sample", feature)
    assert graph["segments"][0]["id"] == "s0"
    assert len(graph["persons"]) == 2

    prediction = tmp_path / "student.json"
    prediction.write_text(json.dumps({
        "task": "ntu120_xsub",
        "topk": [{"label": "run", "score": 0.6}, {"label": "other", "score": 0.4}],
    }), encoding="utf-8")
    task, hint = load_student_hint(prediction, ("walk", "run"))
    assert task == "ntu120_xsub"
    assert hint == {"run": 0.6}


def test_extract_json_object_handles_fence_and_rejects_text():
    assert extract_json_object('prefix ```json\n{"a": 1}\n``` suffix') == {"a": 1}
    assert extract_json_object(
        '```json\n{"a": {"nested": true}, "items": [{"value": 1}]}\n```'
    ) == {"a": {"nested": True}, "items": [{"value": 1}]}
    with pytest.raises(ValueError):
        extract_json_object("no object")


def test_compatibility_teacher_is_lazy(monkeypatch, tmp_path):
    monkeypatch.setenv("DAHUA_MODEL_ROOT", str(tmp_path))
    backend = QwenTeacher()
    assert backend.model is None
    assert backend.config.model_dir.endswith("Qwen3-VL-32B-Instruct")


def test_distribution_similarity_is_bounded_and_symmetric():
    left, right = distribution("normal_walk"), distribution("conflict_push")
    assert distribution_similarity(left, left) == pytest.approx(1.0)
    assert distribution_similarity(left, right) == pytest.approx(distribution_similarity(right, left))
    assert 0 <= distribution_similarity(left, right) < 1


def test_filter_accepts_strong_consistent_signal():
    output = teacher(probability=0.99)
    decision = PseudoLabelFilter().evaluate(
        [output, output], distribution("normal_walk", 0.99), 0.99, 0.99, {}
    )
    assert decision.status == "accepted"
    assert decision.score >= 0.85


def test_filter_forces_review_on_push_without_contact():
    output = teacher("conflict_push", 0.9)
    decision = PseudoLabelFilter().evaluate(
        [output], distribution("conflict_push", 0.9), 0.9, 0.9, {"close_contact": False}
    )
    assert decision.status == "review"
    assert "push_without_contact_evidence" in decision.conflicts
