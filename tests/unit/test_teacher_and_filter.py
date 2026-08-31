import pytest

from dahua_cup.semantic_teacher.api.qwen3_backend import (
    Qwen3Config,
    extract_json_object,
    parse_teacher_output,
)
from dahua_cup.semantic_teacher.prompts.prompt_builder import build_teacher_prompt
from dahua_cup.semantic_teacher.pseudo_label.filter_label import (
    PseudoLabelFilter,
    distribution_similarity,
)
from dahua_cup.semantic_teacher.schemas import Evidence, LABELS, TeacherOutput


def distribution(label="normal_walk", probability=0.9):
    rest = (1 - probability) / (len(LABELS) - 1)
    return {item: probability if item == label else rest for item in LABELS}


def teacher(label="normal_walk", probability=0.9):
    return TeacherOutput("sample", label, distribution(label, probability), probability,
                         evidence=[Evidence("trajectory", "s0", "measured motion")],
                         reason="measured evidence", needs_review=False)


def test_prompt_is_closed_set_and_evidence_based():
    prompt = build_teacher_prompt("sample", {"segments": [{"id": "s0"}]})
    assert "campus6-v1.3" in prompt
    assert "Do not invent" in prompt
    assert "normal_walk" in prompt
    assert "Do not output label" in prompt
    assert "at least one item" in prompt
    assert "student_distribution" not in prompt
    assert "two or more people" in prompt


def test_teacher_response_requires_evidence():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample",
        "distribution": distribution(), "confidence": 0.9,
        "evidence": [], "counter_evidence": [], "reason": "no cue",
        "needs_review": True,
    }
    with pytest.raises(ValueError, match="at least one evidence"):
        parse_teacher_output(value, LABELS)


def test_teacher_response_is_normalized_to_closed_set():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample", "label": "normal_run",
        "distribution": [{"label": "normal_run", "probability": 0.8}, {"label": "normal_walk", "probability": 0.2}],
        "confidence": 0.8, "evidence": [{"type":"trajectory", "segment_id":"s0", "description":"visible motion"}], "counter_evidence": [], "reason": "visible motion", "needs_review": False,
    }
    output, raw = parse_teacher_output(value, LABELS)
    assert raw is None
    assert output.label == "normal_run"
    assert set(output.distribution) == set(LABELS)


def test_teacher_evidence_allows_additive_model_specific_fields():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample",
        "label": "normal_walk", "distribution": distribution(),
        "confidence": 0.9,
        "evidence": [{
            "type": "trajectory", "segment_id": "s0",
            "description": "low measured speed", "pose_coverage": 0.82,
        }],
        "counter_evidence": [], "reason": "measured evidence",
        "needs_review": False,
    }

    output, raw = parse_teacher_output(value, LABELS)

    assert raw is None
    assert output.evidence[0].description == "low measured speed"


def test_teacher_label_is_derived_when_model_omits_it():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample",
        "distribution": [{"label": "normal_run", "probability": 0.8},
                         {"label": "normal_walk", "probability": 0.2}],
        "confidence": 0.8, "evidence": [{"type":"trajectory", "segment_id":"s0", "description":"visible motion"}], "counter_evidence": [],
        "reason": "visible motion", "needs_review": False,
    }
    output, raw = parse_teacher_output(value, LABELS)
    assert raw is None
    assert output.label == "normal_run"


def test_teacher_conflicting_label_is_replaced_by_distribution_argmax():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample",
        "label": "playful_push",
        "distribution": distribution("conflict_push", 0.9),
        "confidence": 0.9, "evidence": [{"type":"trajectory", "segment_id":"s0", "description":"visible motion"}], "counter_evidence": [],
        "reason": "escalating contact", "needs_review": False,
    }
    output, raw = parse_teacher_output(value, LABELS)
    assert output.label == "conflict_push"
    assert raw is not None
    assert raw["raw_label"] == "playful_push"


def test_teacher_schema_version_shorthand_is_normalized():
    value = {
        "schema_version": "v1", "sample_id": "sample",
        "distribution": distribution("normal_walk", 0.9),
        "confidence": 0.9, "evidence": [{"type":"trajectory", "segment_id":"s0", "description":"visible motion"}], "counter_evidence": [],
        "reason": "slow gait", "needs_review": False,
    }
    output, raw = parse_teacher_output(value, LABELS)
    assert output.schema_version == "teacher_output.v1"
    assert raw is not None
    assert raw["schema_version_normalized_from"] == "v1"


def test_qwen_config_does_not_load_weights_on_construction():
    config = Qwen3Config(model_dir="remote-or-server-model", local_files_only=True)
    assert config.model_dir == "remote-or-server-model"
    assert config.attn_implementation == "sdpa"


def test_json_parser_and_filter_contracts():
    assert extract_json_object('```json\n{"label": "normal_walk"}\n```') == {"label": "normal_walk"}
    with pytest.raises(ValueError):
        extract_json_object("not json")
    accepted = PseudoLabelFilter().evaluate([teacher( probability=0.99)] * 2, distribution(probability=0.99), 0.99, 0.99, {})
    assert accepted.status == "accepted"
    assert distribution_similarity(distribution(), distribution()) == pytest.approx(1.0)
