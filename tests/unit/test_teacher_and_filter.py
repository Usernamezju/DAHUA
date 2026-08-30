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
from dahua_cup.semantic_teacher.schemas import LABELS, TeacherOutput


def distribution(label="normal_walk", probability=0.9):
    rest = (1 - probability) / (len(LABELS) - 1)
    return {item: probability if item == label else rest for item in LABELS}


def teacher(label="normal_walk", probability=0.9):
    return TeacherOutput("sample", label, distribution(label, probability), probability,
                         reason="measured evidence", needs_review=False)


def test_prompt_is_closed_set_and_evidence_based():
    prompt = build_teacher_prompt("sample", {"segments": [{"id": "s0"}]}, distribution())
    assert "campus6-v1.0" in prompt
    assert "Do not invent" in prompt
    assert "normal_walk" in prompt


def test_teacher_response_is_normalized_to_closed_set():
    value = {
        "schema_version": "teacher_output.v1", "sample_id": "sample", "label": "normal_run",
        "distribution": [{"label": "normal_run", "probability": 0.8}, {"label": "normal_walk", "probability": 0.2}],
        "confidence": 0.8, "evidence": [], "counter_evidence": [], "reason": "visible motion", "needs_review": False,
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
