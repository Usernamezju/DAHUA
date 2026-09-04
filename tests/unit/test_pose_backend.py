import json

import pytest

from dahua_cup.backend.config import Settings
from dahua_cup.backend.pose_backend import (
    DEFAULT_POSE_BACKEND,
    load_pose_backend,
    pose_backend_status,
    save_pose_backend,
)


def _write_engine_set(root):
    for name in ("rtmdet_s", "rtmpose_s"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "deploy.json").write_text("{}", encoding="utf-8")
        (directory / "end2end.engine").write_bytes(b"engine")
    (root / "validation.json").write_text(
        json.dumps({
            "status": "passed",
            "detector": {"status": "passed"},
            "pose": {"status": "passed"},
        }),
        encoding="utf-8",
    )


def _write_fp16_engine_set(root):
    for name in ("rtmdet_nano_person", "rtmpose_s"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "deploy.json").write_text("{}", encoding="utf-8")
        (directory / "end2end.engine").write_bytes(b"engine")
    (root / "validation.json").write_text(
        json.dumps({
            "status": "passed",
            "detector": {"status": "passed"},
            "pose": {"status": "passed"},
        }),
        encoding="utf-8",
    )


def test_defaults_to_fp16_and_reports_when_engines_are_not_deployed(tmp_path):
    config = tmp_path / "settings" / "pose_backend.json"
    status = pose_backend_status(config, tmp_path / "int8", fp32_available=True)

    assert load_pose_backend(config) == DEFAULT_POSE_BACKEND
    assert status["selected"] == "tensorrt_fp16"
    assert [item["available"] for item in status["choices"]] == [True, False, False]


def test_environment_backend_overrides_persisted_selection(tmp_path, monkeypatch):
    config = tmp_path / "settings" / "pose_backend.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"selected": "mmpose_fp32"}), encoding="utf-8")

    monkeypatch.setenv("DAHUA_POSE_BACKEND", "tensorrt_fp16")
    assert load_pose_backend(config) == "tensorrt_fp16"


def test_int8_can_only_be_selected_after_both_engine_sets_exist(tmp_path):
    config, root = tmp_path / "settings" / "pose_backend.json", tmp_path / "int8"
    with pytest.raises(ValueError, match="尚不可用"):
        save_pose_backend(config, "tensorrt_int8", root, fp32_available=True)

    _write_engine_set(root)
    status = save_pose_backend(config, "tensorrt_int8", root, fp32_available=True)

    assert status["selected"] == "tensorrt_int8"
    assert status["selected_available"] is True
    assert json.loads(config.read_text(encoding="utf-8")) == {"selected": "tensorrt_int8"}


def test_fp16_can_only_be_selected_after_verified_nano_and_pose_engines(tmp_path):
    config, root = tmp_path / "settings" / "pose_backend.json", tmp_path / "fp16"
    with pytest.raises(ValueError, match="尚不可用"):
        save_pose_backend(
            config, "tensorrt_fp16", tmp_path / "int8", fp32_available=True, fp16_root=root
        )

    _write_fp16_engine_set(root)
    status = save_pose_backend(
        config, "tensorrt_fp16", tmp_path / "int8", fp32_available=True, fp16_root=root
    )

    assert status["selected"] == "tensorrt_fp16"
    assert status["selected_available"] is True


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="未知"):
        save_pose_backend(tmp_path / "pose_backend.json", "other", tmp_path / "int8", fp32_available=True)


def test_settings_switches_only_future_pose_command_to_ready_backend(tmp_path, monkeypatch):
    python = tmp_path / "rtmpose-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    settings = Settings(
        repository_root=tmp_path,
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        source_root=tmp_path / "runtime" / "videos",
        manifest_path=tmp_path / "runtime" / "manifest.csv",
        database_path=tmp_path / "runtime" / "review.sqlite3",
        artifact_root=tmp_path / "runtime" / "artifacts",
        frontend_root=tmp_path / "frontend",
        pose_command="",
        student_command="",
        ffmpeg=None,
    )
    monkeypatch.setenv("DAHUA_RTMPOSE_PYTHON", str(python))

    # Production defaults to FP16 and remains unavailable until both verified
    # engines are deployed; FP32 can still be selected deliberately for diagnosis.
    assert settings.default_pose_command() == ""
    settings.update_pose_backend("mmpose_fp32")
    assert "--backend mmpose" in settings.default_pose_command()
    _write_engine_set(settings.pose_int8_model_root)
    settings.update_pose_backend("tensorrt_int8")
    command = settings.default_pose_command()
    assert "--backend mmdeploy" in command
    assert str(settings.pose_int8_model_root / "rtmdet_s") in command

    _write_fp16_engine_set(settings.pose_fp16_model_root)
    settings.update_pose_backend("tensorrt_fp16")
    command = settings.default_pose_command()
    assert "--backend mmdeploy" in command
    assert str(settings.pose_fp16_model_root / "rtmdet_nano_person") in command
