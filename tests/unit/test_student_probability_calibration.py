import json

import numpy as np
import pytest

from dahua_cup.backend.config import Settings
from dahua_cup.pipeline.rtmpose17_student_worker import (
    retarget_temperature_probabilities,
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


def test_retarget_temperature_softens_a_cached_distribution_without_reordering():
    cached = np.array([0.998, 0.0008, 0.0007, 0.0002, 0.0002, 0.0001])

    softened = retarget_temperature_probabilities(cached, 1.79725, 5.0)

    assert softened.sum() == pytest.approx(1.0)
    assert softened.argmax() == cached.argmax()
    assert softened[0] < cached[0]
    assert softened[1] > cached[1]


def test_settings_defaults_to_review_temperature_and_supports_override(
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
    monkeypatch.delenv("DAHUA_CAMPUS6_REVIEW_TEMPERATURE", raising=False)

    assert Settings.from_env().student_probability_temperature() == pytest.approx(5.0)
    monkeypatch.setenv("DAHUA_CAMPUS6_REVIEW_TEMPERATURE", "4.0")
    assert Settings.from_env().student_probability_temperature() == pytest.approx(4.0)
