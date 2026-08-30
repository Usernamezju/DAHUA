import json
import pickle

import numpy as np

from dahua_cup.backend.baseline import Campus6Baseline


def test_prediction_rows_follow_annotation_order_without_changing_sample_ids(tmp_path):
    annotations = tmp_path / "annotations.pkl"
    predictions = tmp_path / "predictions.pkl"
    rows = [
        {
            "frame_dir": "normal_walk/a",
            "label": 0,
            "keypoint": np.zeros((1, 2, 17, 2), dtype=np.float32),
            "keypoint_score": np.ones((1, 2, 17), dtype=np.float32),
        },
        {
            "frame_dir": "normal_run/b",
            "label": 1,
            "keypoint": np.zeros((1, 2, 17, 2), dtype=np.float32),
            "keypoint_score": np.ones((1, 2, 17), dtype=np.float32),
        },
    ]
    annotations.write_bytes(pickle.dumps({
        "annotations": rows,
        "split": {"all": ["normal_run/b", "normal_walk/a"]},
    }))
    probabilities = np.asarray([
        [0.9, 0.1, 0, 0, 0, 0],
        [0.1, 0.9, 0, 0, 0, 0],
    ], dtype=np.float32)
    predictions.write_bytes(pickle.dumps(probabilities))
    predictions.with_suffix(".pkl.json").write_text(
        json.dumps({
            "order": "annotations",
            "confidence_calibration": {"temperature": 1.79725},
        }),
        encoding="utf-8",
    )

    baseline = Campus6Baseline(annotations, predictions, review_temperature=5.0)
    first, second = baseline.sample_ids()
    assert baseline.prediction(first)["topk"][0]["label"] == "normal_run"
    assert baseline.prediction(second)["topk"][0]["label"] == "normal_walk"
    assert baseline.prediction(first)["topk"][0]["score"] < 0.9
    assert baseline.prediction(first)["confidence_calibration"] == {
        "method": "temperature_scaling",
        "temperature": 5.0,
        "validation_temperature": 1.79725,
        "review_softening": True,
        "top1_preserved": True,
    }
    assert baseline.evaluation_summary()["overall_accuracy"] == 1.0
