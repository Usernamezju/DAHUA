"""Unit tests for the runtime-adjustable macro parameter surface."""

from pathlib import Path

import pytest

from dahua_cup.backend.config import Settings
from dahua_cup.backend.macro_parameters import (
    ENV_OVERRIDES,
    MACRO_PARAMETERS,
    apply_values,
    sanitize_values,
    snapshot,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        repository_root=tmp_path,
        data_root=tmp_path,
        runtime_root=tmp_path,
        source_root=tmp_path / "videos",
        manifest_path=tmp_path / "campus6_manifest.csv",
        database_path=tmp_path / "review.sqlite3",
        artifact_root=tmp_path / "artifacts",
        frontend_root=tmp_path,
        pose_command="",
        student_command="",
        ffmpeg=None,
    )


def test_incremental_batch_size_is_registered():
    spec = {item["key"]: item for item in MACRO_PARAMETERS}["incremental_batch_size"]
    assert spec["type"] == "nonnegative_int"
    assert spec["default"] == 50
    assert spec["min"] == 1
    assert spec["group"] == "incremental"
    assert "incremental_batch_size" in ENV_OVERRIDES


def test_incremental_batch_size_requires_positive_integer():
    with pytest.raises(ValueError):
        sanitize_values({"incremental_batch_size": 0}, strict=True)
    with pytest.raises(ValueError):
        sanitize_values({"incremental_batch_size": 12.5}, strict=True)
    assert sanitize_values({"incremental_batch_size": 80}, strict=True) == {
        "incremental_batch_size": 80
    }


def test_apply_and_snapshot_incremental_batch_size(tmp_path):
    settings = _settings(tmp_path)
    cleaned = apply_values(settings, {"incremental_batch_size": 30})
    assert cleaned == {"incremental_batch_size": 30}
    assert settings.incremental_batch_size == 30
    report = snapshot(settings)
    by_key = {item["key"]: item for item in report["parameters"]}
    assert by_key["incremental_batch_size"]["value"] == 30
    assert "incremental" in report["groups"]


def test_apply_values_rejects_unknown_keys(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError):
        apply_values(settings, {"incremental_batch_size": 25, "not_a_parameter": 1})
