import json

import pytest

from dahua_cup.evaluation.metrics import (
    evaluate_records,
    render_markdown_report,
)
from dahua_cup.pipeline.evaluate_predictions import main


LABELS = ("normal_walk", "normal_run")


def _records():
    return [
        {
            "sample_id": "a",
            "true_label": "normal_walk",
            "predicted_label": "normal_walk",
            "distribution": {"normal_walk": 0.8, "normal_run": 0.2},
            "latency_ms": 10,
            "slices": {"occlusion": "none", "lighting": "day"},
        },
        {
            "sample_id": "b",
            "true_label": "normal_walk",
            "predicted_label": "normal_run",
            "confidence": 0.6,
            "latency_ms": 20,
            "slices": {"occlusion": "partial", "lighting": "day"},
        },
        {
            "sample_id": "c",
            "true_label": "normal_run",
            "predicted_label": "normal_run",
            "distribution": {"normal_walk": 0.1, "normal_run": 0.9},
            "latency_ms": 30,
            "slices": {"occlusion": "none", "lighting": "night"},
        },
        {
            "sample_id": "d",
            "true_label": "normal_run",
            "predicted_label": "normal_walk",
            "topk": [{"label": "normal_walk", "score": 0.55}],
            "latency_ms": 40,
            "slices": {"occlusion": "partial", "lighting": "night"},
        },
    ]


def test_evaluation_reports_classification_calibration_slices_and_bundle(tmp_path):
    first = tmp_path / "student.pth"
    second = tmp_path / "pose.task"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 20)
    report = evaluate_records(
        _records(),
        LABELS,
        ece_bins=5,
        edge_models=[first, second],
        maximum_edge_bytes=30,
    )

    assert report["classification"]["accuracy"] == 0.5
    assert report["classification"]["macro_f1"] == 0.5
    assert report["classification"]["confusion_matrix"]["values"] == [
        [1, 1],
        [1, 1],
    ]
    assert report["calibration"]["confidence_sample_count"] == 4
    assert report["calibration"]["full_distribution_sample_count"] == 2
    assert report["latency"]["p50_ms"] == 25
    assert report["robustness_slices"]["occlusion"]["none"]["accuracy"] == 1
    assert report["edge_model_bundle"]["passed"]
    assert "混淆矩阵" in render_markdown_report(report)


def test_evaluation_rejects_duplicate_samples():
    rows = _records()
    rows[1]["sample_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_records(rows, LABELS)


def test_evaluation_cli_writes_json_and_markdown(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _records()),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    main(
        [
            "--input-jsonl",
            str(input_path),
            "--output",
            str(output),
            "--labels",
            ",".join(LABELS),
        ]
    )
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        "competition_evaluation.v1"
    )
