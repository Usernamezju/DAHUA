import json

import pytest

from dahua_cup.semantic_teacher.incremental.hard_mining import (
    hard_sample_score,
    normalized_entropy,
)
from dahua_cup.semantic_teacher.incremental.model_registry import ModelRegistry
from dahua_cup.semantic_teacher.incremental.replay_buffer import build_replay_buffer
from dahua_cup.semantic_teacher.pseudo_label.review_queue import ReviewQueue
from dahua_cup.semantic_teacher.schemas import LABELS


def test_review_queue_records_review_and_event(tmp_path):
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    queue.enqueue("s1", "artifact.npz", "normal_walk", 0.7, {"prompt": "v1"}, priority=2)
    assert queue.pending()[0]["sample_id"] == "s1"
    queue.submit_review("s1", "reviewer", "normal_run", "wrong_speed")
    assert queue.pending() == []
    with pytest.raises(ValueError, match="already"):
        queue.submit_review("s1", "reviewer", "normal_run", "duplicate")


def test_replay_buffer_is_balanced_and_deterministic():
    samples = [{"sample_id": f"a{i}", "label": "a", "scene": "x", "camera": "1"} for i in range(5)]
    samples += [{"sample_id": f"b{i}", "label": "b", "scene": "x", "camera": "1"} for i in range(2)]
    first = build_replay_buffer(samples, 4)
    second = build_replay_buffer(reversed(samples), 4)
    assert first == second
    assert {item["label"] for item in first} == {"a", "b"}


def test_hard_sample_entropy_and_disagreement():
    uniform = {label: 1 / len(LABELS) for label in LABELS}
    certain = {label: float(label == "normal_walk") for label in LABELS}
    assert normalized_entropy(uniform) == pytest.approx(1.0)
    assert hard_sample_score(uniform) > hard_sample_score(certain)


def test_model_registry_guards_transitions(tmp_path):
    registry = ModelRegistry(tmp_path / "models.jsonl")
    record = registry.register({
        "model_id": "m1", "dataset_id": "d1", "config_hash": "a", "checkpoint_hash": "b",
        "metrics": {}, "edge_size_bytes": 10, "latency": {},
    })
    assert record["status"] == "candidate"
    with pytest.raises(ValueError, match="invalid"):
        registry.transition("m1", "production")
    assert registry.transition("m1", "validated")["status"] == "validated"
