import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dahua_cup.backend.config import Settings
from dahua_cup.backend.evolution import (
    EvolutionConfiguration,
    evaluate_ntu120_release,
)
from dahua_cup.pipeline.build_ntu120_distill_dataset import build_dataset
from dahua_cup.pipeline.collect_ntu120_pseudo import (
    NTU120PseudoThresholds,
    make_record,
    merge_records,
)
from dahua_cup.tests.unit.test_web_application import make_settings


def labels():
    return tuple(f"class_{index}" for index in range(120))


def pose_feature(path: Path) -> None:
    np.savez_compressed(
        path,
        keypoint=np.ones((2, 8, 25, 3), dtype=np.float32),
    )


def prediction(feature: Path, *, label="class_7"):
    return {
        "sample_id": "sample-1",
        "feature": str(feature),
        "checkpoint": "/models/original.pth",
        "checkpoint_sha256": "old-hash",
        "topk": [
            {"label": label, "score": 0.72},
            {"label": "class_3", "score": 0.18},
        ],
        "teacher_gate": {
            "pose_metrics": {"quality": 0.95},
            "instability": {"score": 0.05},
        },
    }


def teacher_distribution(label="class_7"):
    value = {name: 0.05 / 119 for name in labels()}
    value[label] = 0.95
    return value


def teacher(*, label="class_7", needs_review=False):
    return {
        "model": "Qwen3-VL",
        "result": {
            "sample_id": "sample-1",
            "label": label,
            "distribution": teacher_distribution(label),
            "confidence": 0.95,
            "needs_review": needs_review,
        },
    }


def test_ntu120_pseudo_filter_accepts_only_trainable_feature(tmp_path):
    feature = tmp_path / "sample.npz"
    pose_feature(feature)
    accepted = make_record(
        prediction(feature),
        teacher(),
        labels(),
        NTU120PseudoThresholds(),
    )
    assert accepted["status"] == "accepted"
    assert accepted["label_index"] == 7
    assert len(accepted["soft_label"]) == 120
    assert sum(accepted["soft_label"]) == pytest.approx(1.0)
    assert accepted["confidence_method"] == "qwen_self_report_uncalibrated"

    missing = make_record(
        prediction(tmp_path / "missing.npz"),
        teacher(),
        labels(),
        NTU120PseudoThresholds(),
    )
    assert missing["status"] != "accepted"
    assert "pose_feature_missing" in missing["review_reasons"]


def test_ntu120_student_teacher_conflict_enters_review(tmp_path):
    feature = tmp_path / "sample.npz"
    pose_feature(feature)
    value = make_record(
        prediction(feature, label="class_2"),
        teacher(label="class_7", needs_review=True),
        labels(),
        NTU120PseudoThresholds(),
    )
    assert value["status"] == "review"
    assert "teacher_requested_review" in value["review_reasons"]


def test_low_pose_quality_is_weighted_but_not_a_review_veto(tmp_path):
    feature = tmp_path / "sample.npz"
    pose_feature(feature)
    student = prediction(feature)
    student["teacher_gate"]["pose_metrics"]["quality"] = 0.40
    value = make_record(
        student,
        teacher(),
        labels(),
        NTU120PseudoThresholds(),
    )

    assert value["status"] == "accepted"
    assert value["review_reasons"] == []
    assert "pose_quality_below_threshold" in value["quality_warnings"]


def test_ntu120_pseudo_records_keep_monotonic_versions(tmp_path):
    feature = tmp_path / "sample.npz"
    pose_feature(feature)
    first = make_record(
        prediction(feature),
        teacher(),
        labels(),
        NTU120PseudoThresholds(),
    )
    version_one = merge_records([], [first])[0]
    changed = dict(first, fingerprint="new-fingerprint", quality_score=0.8)
    version_two = merge_records([version_one], [changed])[0]
    assert version_one["version"] == 1
    assert version_two["version"] == 2
    assert version_two["supersedes_fingerprint"] == first["fingerprint"]


def test_ntu120_distill_dataset_combines_replay_pseudo_and_validation(tmp_path):
    feature = tmp_path / "sample.npz"
    pose_feature(feature)
    pseudo = make_record(
        prediction(feature),
        teacher(),
        labels(),
        NTU120PseudoThresholds(),
    )
    train_annotations = [
        {
            "frame_dir": f"train-{class_index}",
            "label": class_index,
            "keypoint": np.ones((2, 4, 25, 3), dtype=np.float32),
            "total_frames": 4,
        }
        for class_index in range(120)
    ]
    base = {
        "split": {
            "xsub_train": [
                row["frame_dir"] for row in train_annotations
            ],
            "xsub_val": ["val-a"],
        },
        "annotations": train_annotations + [
            {
                "frame_dir": "val-a",
                "label": 3,
                "keypoint": np.ones((2, 4, 25, 3), dtype=np.float32),
                "total_frames": 4,
            },
        ],
    }
    data, metadata = build_dataset(
        base,
        [pseudo],
        labels(),
        max_replay_samples=1,
        max_pseudo_samples=1,
    )
    assert metadata["replay_samples"] == 1
    assert metadata["pseudo_samples"] == 1
    assert metadata["validation_samples"] == 1
    pseudo_row = next(
        row for row in data["annotations"]
        if row.get("sample_id") == "sample-1"
    )
    assert pseudo_row["teacher_valid"] == 1
    assert pseudo_row["has_hard_label"] == 0
    assert pseudo_row["teacher_distribution"].shape == (120,)
    assert data["split"]["xsub_val"] == ["val-a"]


def test_ntu120_release_requires_all_quality_and_size_gates():
    baseline = {
        "top1_acc": 0.80,
        "top5_acc": 0.95,
        "mean_class_accuracy": 0.75,
    }
    candidate = {
        "top1_acc": 0.799,
        "top5_acc": 0.96,
        "mean_class_accuracy": 0.752,
    }
    passed = evaluate_ntu120_release(
        candidate,
        baseline,
        candidate_size_bytes=49 * 1024 * 1024,
        maximum_checkpoint_bytes=50 * 1024 * 1024,
        minimum_mean_class_improvement=0.001,
        maximum_top1_drop=0.002,
    )
    assert passed["passed"]

    rejected = evaluate_ntu120_release(
        {**candidate, "top1_acc": 0.79},
        baseline,
        candidate_size_bytes=49 * 1024 * 1024,
        maximum_checkpoint_bytes=50 * 1024 * 1024,
        minimum_mean_class_improvement=0.001,
        maximum_top1_drop=0.002,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["top1_not_regressed"]


def test_promoted_checkpoint_becomes_active_only_through_pointer(tmp_path):
    settings = make_settings(tmp_path)
    base = tmp_path / "base.pth"
    promoted = tmp_path / "promoted.pth"
    base.write_bytes(b"base")
    promoted.write_bytes(b"promoted")
    registry = settings.runtime_root / "evolution" / "models" / "registry.jsonl"
    pointer = settings.runtime_root / "evolution" / "models" / "production.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "model_id": "candidate-1",
                "status": "production",
                "checkpoint_path": str(promoted),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pointer.write_text(
        json.dumps({"current_model_id": "candidate-1"}),
        encoding="utf-8",
    )
    configured = replace(
        settings,
        model_registry_path=registry,
        production_pointer_path=pointer,
    )
    assert configured.resolve_student_checkpoint() == promoted.resolve()


def test_default_auto_evolution_configuration_is_ntu120():
    root = Path(__file__).resolve().parents[3]
    config = EvolutionConfiguration.load(
        root / "dahua_cup/configs/campus/auto_evolution_ntu120.yaml",
        data_root=root,
        repository_root=root,
    )
    assert config.enabled
    assert config.minimum_accepted == 20
    assert config.maximum_checkpoint_bytes == 50 * 1024 * 1024
    assert config.base_annotation_candidates[0] == (
        root / "datasets" / "NTU" / "ProtoGCN" / "ntu120_3danno.pkl"
    ).resolve()
