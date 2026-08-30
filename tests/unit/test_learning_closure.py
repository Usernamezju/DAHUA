import json
import pickle

import numpy as np
import pytest

from dahua_cup.backend.store import ReviewStore
from dahua_cup.pipeline.build_campus6_dataset import build_dataset
from dahua_cup.pipeline.generate_pseudo_dataset import main as pseudo_main
from dahua_cup.pipeline.mine_candidates import (
    class_statistics,
    main as mine_main,
    mine,
    select_budget,
)
from dahua_cup.pipeline.train_campus6 import main as train_campus6_main
from dahua_cup.scripts.prepare_protogcn_campus6_checkpoint import (
    campus6_backbone_state,
)
from dahua_cup.semantic_teacher.incremental.release_gate import (
    ProductionPointer,
    evaluate_release,
)
from dahua_cup.semantic_teacher.pseudo_label.calibration import (
    TemperatureCalibrator,
    fit_temperature,
)
from dahua_cup.semantic_teacher.pseudo_label.filter_label import (
    load_filter_configuration,
)
from dahua_cup.semantic_teacher.schemas import LABELS, TeacherOutput


def distribution(label, probability=0.9):
    remainder = (1.0 - probability) / (len(LABELS) - 1)
    return {
        value: probability if value == label else remainder for value in LABELS
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def pose_feature(path, frames=8):
    np.savez_compressed(
        path,
        keypoint=np.ones((2, frames, 25, 3), dtype=np.float32),
        fps=np.asarray(25.0),
    )


def test_threshold_yaml_is_live_configuration():
    thresholds, weights = load_filter_configuration(
        "dahua_cup/configs/campus/thresholds.yaml"
    )
    assert thresholds.accept_score == 0.85
    assert thresholds.accepted_audit_rate == 0.10
    assert thresholds.rare_class_audit_rate == 0.20
    assert sum(weights.values()) == pytest.approx(1.0)


def test_teacher_temperature_calibration_is_fitted_and_serializable():
    samples = [
        (distribution("normal_walk", 0.99), "normal_run"),
        (distribution("normal_run", 0.99), "normal_walk"),
        (distribution("playful_push", 0.99), "playful_push"),
    ]
    fitted = fit_temperature(samples, steps=24)
    assert fitted.temperature > 1.0
    restored = TemperatureCalibrator.from_dict(fitted.to_dict())
    raw = distribution("normal_walk", 0.99)
    assert restored.transform(raw)["normal_walk"] < raw["normal_walk"]


def test_candidate_mining_ranks_uncertain_and_balances_budget():
    certain = {
        "sample_id": "certain",
        "distribution": distribution("normal_walk", 0.99),
    }
    uncertain = {
        "sample_id": "uncertain",
        "distribution": {label: 1 / len(LABELS) for label in LABELS},
        "novelty": 1.0,
    }
    ranked = mine([certain, uncertain])
    assert ranked[0]["sample_id"] == "uncertain"
    assert ranked[0]["hard_score"] > ranked[1]["hard_score"]
    assert select_budget(ranked, limit=1)[0]["sample_id"] == "uncertain"


def test_ntu120_mining_automatically_detects_rare_classes():
    rows = []
    for index in range(4):
        rows.append(
            {
                "sample_id": f"common-{index}",
                "task": "ntu120_xsub",
                "label_space_size": 120,
                "topk": [
                    {"label": "walking", "score": 0.80},
                    {"label": "running", "score": 0.10},
                ],
            }
        )
    rows.append(
        {
            "sample_id": "rare-1",
            "task": "ntu120_xsub",
            "label_space_size": 120,
            "topk": [
                {"label": "jump up", "score": 0.55},
                {"label": "jump down", "score": 0.35},
            ],
        }
    )
    ranked = mine(rows)
    rare = next(row for row in ranked if row["sample_id"] == "rare-1")
    assert rare["rare_class"]
    assert rare["predicted_class_count"] == 1
    assert rare["rarity_score"] == pytest.approx(0.75)
    assert rare["label_space_size"] == 120
    statistics = class_statistics(
        [row["predicted_label"] for row in ranked]
    )
    assert statistics["rare_labels"] == ["jump up"]


def test_ntu120_mining_reads_web_gate_quality_and_instability():
    ranked = mine(
        [
            {
                "sample_id": "web-hard",
                "task": "ntu120_xsub",
                "label_space_size": 120,
                "topk": [
                    {"label": "walking", "score": 0.55},
                    {"label": "running", "score": 0.40},
                ],
                "teacher_gate": {
                    "pose_metrics": {"quality": 0.30},
                    "instability": {"score": 0.90},
                    "teacher_conflict": {"conflict": True},
                },
            }
        ]
    )
    components = ranked[0]["hard_components"]
    assert components["quality_failure"] == pytest.approx(0.70)
    assert components["temporal_instability"] == pytest.approx(0.90)
    assert components["rule_conflict"] == pytest.approx(1.0)


def test_ntu120_prediction_directory_can_enqueue_web_hard_cases(
    tmp_path, capsys
):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    database = tmp_path / "review.sqlite3"
    store = ReviewStore(database)
    for sample_id, first, second in (
        ("certain", 0.95, 0.02),
        ("uncertain", 0.36, 0.34),
    ):
        video = tmp_path / f"{sample_id}.mp4"
        video.write_bytes(b"video")
        store.add_sample(sample_id, video)
        (prediction_dir / f"{sample_id}.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "task": "ntu120_xsub",
                    "label_space_size": 120,
                    "topk": [
                        {"label": "walking", "score": first},
                        {"label": "running", "score": second},
                    ],
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "hard.jsonl"
    statistics = tmp_path / "stats.json"
    mine_main(
        [
            "--prediction-dir",
            str(prediction_dir),
            "--output",
            str(output),
            "--stats-output",
            str(statistics),
            "--review-db",
            str(database),
            "--minimum-score",
            "0",
            "--limit",
            "1",
        ]
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["sample_id"] == "uncertain"
    queued = store.get_sample("uncertain")
    assert queued["hard_score"] == pytest.approx(selected["hard_score"])
    assert queued["priority"] == selected["review_priority"]
    assert json.loads(statistics.read_text())["label_space_sizes"] == [120]
    assert "candidate_mining_complete" in capsys.readouterr().out


def test_filter_to_web_review_and_jsonl_writeback(tmp_path):
    teacher_path = tmp_path / "teacher.jsonl"
    student_path = tmp_path / "student.jsonl"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    teacher = TeacherOutput(
        "s1",
        "conflict_push",
        distribution("conflict_push", 0.95),
        0.95,
    )
    write_jsonl(teacher_path, [teacher.to_dict()])
    write_jsonl(
        student_path,
        [
            {
                "sample_id": "s1",
                "distribution": distribution("conflict_push", 0.95),
                "pose_quality": 0.95,
                "temporal_stability": 0.95,
                "evidence_flags": {"close_contact": False},
                "video_path": str(video),
                "hard_score": 0.8,
            }
        ],
    )
    database = tmp_path / "review.sqlite3"
    pseudo_main(
        [
            "--teacher-jsonl",
            str(teacher_path),
            "--student-jsonl",
            str(student_path),
            "--output-dir",
            str(tmp_path / "pseudo"),
            "--dataset-version",
            "v1",
            "--prompt-version",
            "p1",
            "--review-db",
            str(database),
        ]
    )
    pseudo_path = tmp_path / "pseudo" / "v1" / "pseudo_labels.jsonl"
    rows = [json.loads(line) for line in pseudo_path.read_text().splitlines()]
    assert rows[0]["status"] == "review"

    store = ReviewStore(database)
    queued = store.list_samples(status="pending")["items"]
    assert queued[0]["sample_id"] == "s1"
    assert queued[0]["quality_score"] > 0
    store.submit_review(
        "s1", "alice", "playful_push", "intent_playful", "confirmed"
    )
    reviewed = json.loads(pseudo_path.read_text().strip())
    assert reviewed["status"] == "accepted"
    assert reviewed["label"] == "playful_push"
    assert reviewed["review"]["reviewer"] == "alice"
    assert reviewed["teacher_soft_label"]

    original_teacher_distribution = reviewed["teacher_soft_label"]
    store.submit_review(
        "s1", "bob", "normal_run", "second_opinion", "changed"
    )
    changed = json.loads(pseudo_path.read_text().strip())
    assert changed["label"] == "normal_run"
    assert changed["teacher_soft_label"] == original_teacher_distribution
    assert changed["review_history"][-1]["reviewer"] == "alice"

    restored = store.undo_last_review("s1", "bob")
    restored_pseudo = json.loads(pseudo_path.read_text().strip())
    assert restored_pseudo["label"] == "playful_push"
    assert restored_pseudo["review"]["reviewer"] == "alice"
    assert restored["reviewer"] == "alice"

    store.undo_last_review("s1", "alice")
    reverted = json.loads(pseudo_path.read_text().strip())
    assert reverted["status"] == "review"
    assert reverted["label"] == "conflict_push"
    assert reverted["review"] is None


def test_build_campus6_dataset_injects_pseudo_and_previous_distribution(tmp_path):
    human_feature = tmp_path / "human.npz"
    pseudo_feature = tmp_path / "pseudo.npz"
    pose_feature(human_feature)
    pose_feature(pseudo_feature)
    sample_rows = [
        {
            "sample_id": "human",
            "feature_path": str(human_feature),
            "label": "normal_walk",
            "label_source": "human",
            "split": "val",
            "source_hash": "human-source",
        },
        {
            "sample_id": "pseudo",
            "feature_path": str(pseudo_feature),
            "split": "train",
            "source_hash": "pseudo-source",
        },
    ]
    pseudo_rows = [
        {
            "sample_id": "pseudo",
            "status": "accepted",
            "label": "playful_chase",
            "soft_label": list(distribution("playful_chase", 0.9).values()),
            "quality_score": 0.88,
            "source": "qwen_teacher",
            "review": None,
        }
    ]
    previous_rows = [
        {
            "sample_id": "pseudo",
            "distribution": distribution("playful_chase", 0.75),
        }
    ]
    data, manifest = build_dataset(
        sample_rows,
        pseudo_rows=pseudo_rows,
        previous_rows=previous_rows,
    )
    assert data["split"] == {
        "train": ["pseudo"],
        "val": ["human"],
        "test": [],
    }
    pseudo = next(
        row for row in data["annotations"] if row["sample_id"] == "pseudo"
    )
    assert pseudo["keypoint"].shape == (2, 8, 25, 3)
    assert pseudo["has_hard_label"] == 0
    assert pseudo["teacher_valid"] == 1
    assert pseudo["previous_valid"] == 1
    assert next(row for row in manifest if row["sample_id"] == "pseudo")[
        "quality_weight"
    ] == pytest.approx(0.88)


def test_build_campus6_dataset_injects_new_pseudo_sample(tmp_path):
    pseudo_feature = tmp_path / "new-pseudo.npz"
    pose_feature(pseudo_feature)
    data, manifest = build_dataset(
        [],
        pseudo_rows=[
            {
                "sample_id": "new-pseudo",
                "feature_path": str(pseudo_feature),
                "source_hash": "new-pseudo-source",
                "status": "accepted",
                "label": "normal_run",
                "soft_label": list(distribution("normal_run").values()),
                "quality_score": 0.9,
                "source": "qwen_teacher",
            }
        ],
    )
    assert data["split"]["train"] == ["new-pseudo"]
    assert manifest[0]["label_source"] == "pseudo"


def test_checkpoint_conversion_strips_ddp_and_class_head():
    state, removed = campus6_backbone_state(
        {
            "module.backbone.layer.weight": "backbone",
            "module.cls_head.fc_cls.weight": "classes",
            "module.cls_head.csc_loss.cl_fc.weight": "contrastive",
        }
    )
    assert state == {"backbone.layer.weight": "backbone"}
    assert removed == [
        "module.cls_head.csc_loss.cl_fc.weight",
        "module.cls_head.fc_cls.weight",
    ]


def test_dataset_rejects_source_leakage(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    pose_feature(first)
    pose_feature(second)
    with pytest.raises(ValueError, match="leaks"):
        build_dataset(
            [
                {
                    "sample_id": "a",
                    "feature_path": str(first),
                    "label": "normal_walk",
                    "split": "train",
                    "source_hash": "same-video",
                },
                {
                    "sample_id": "b",
                    "feature_path": str(second),
                    "label": "normal_walk",
                    "split": "val",
                    "source_hash": "same-video",
                },
            ]
        )


def test_release_gate_and_production_rollback(tmp_path):
    baseline = {
        "global_macro_f1": 0.70,
        "new_macro_f1": 0.60,
        "old_macro_f1": 0.72,
        "dangerous_recall": 0.68,
        "ece": 0.08,
    }
    candidate = {
        "global_macro_f1": 0.72,
        "new_macro_f1": 0.66,
        "old_macro_f1": 0.71,
        "dangerous_recall": 0.70,
        "ece": 0.07,
        "edge_size_bytes": 10,
    }
    report = evaluate_release(candidate, baseline)
    assert report["passed"]
    pointer = ProductionPointer(tmp_path / "production.json")
    pointer.promote("model-v1", report)
    pointer.promote("model-v2", report)
    assert pointer.read()["current_model_id"] == "model-v2"
    assert pointer.rollback("alice", "regression")["current_model_id"] == "model-v1"


def test_train_campus6_dry_run_uses_project_config(tmp_path, capsys):
    annotation = tmp_path / "annotations.pkl"
    with annotation.open("wb") as stream:
        pickle.dump({"split": {}, "annotations": []}, stream)
    init = tmp_path / "init.pth"
    init.write_bytes(b"checkpoint")
    train_campus6_main(
        [
            "--ann-file",
            str(annotation),
            "--work-dir",
            str(tmp_path / "work"),
            "--init-checkpoint",
            str(init),
            "--distill",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "campus6_ntu25_bone_distill.py" in output
    assert "DAHUA_PROTOGCN_CAMPUS6_INIT" in output
