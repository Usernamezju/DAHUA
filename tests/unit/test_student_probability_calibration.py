import json

import numpy as np
import pytest

from dahua_cup.backend.config import Settings
from dahua_cup.pipeline.rtmpose17_student_worker import (
    temperature_scale_probabilities,
)


def test_temperature_scaling_softens_overconfident_int8_probabilities():
    raw = np.array([0.985, 0.013, 0.001, 0.0005, 0.0003, 0.0002])

    calibrated = temperature_scale_probabilities(raw, 1.7972517356410385)

    assert calibrated.sum() == pytest.approx(1.0)
    assert calibrated.argmax() == raw.argmax()
    assert calibrated[0] < raw[0]
    assert calibrated[1] > raw[1]


def test_identity_temperature_does_not_change_probabilities():
    raw = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])

    assert np.allclose(temperature_scale_probabilities(raw, 1.0), raw)


def test_settings_reads_temperature_from_accepted_prediction_sidecar(
    tmp_path, monkeypatch
):
    predictions = tmp_path / "M1KD.eval_all.pkl"
    predictions.write_bytes(b"cache")
    predictions.with_suffix(".pkl.json").write_text(
        json.dumps({"confidence_calibration": {"temperature": 1.79725}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DAHUA_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("DAHUA_DATA_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("DAHUA_CAMPUS6_BASELINE_PREDICTIONS", str(predictions))
    monkeypatch.delenv("DAHUA_CAMPUS6_PROBABILITY_TEMPERATURE", raising=False)

    assert Settings.from_env().student_probability_temperature() == pytest.approx(
        1.79725
    )
