import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dahua_cup.backend import qwen_api
from dahua_cup.backend.jobs import JobManager


def _teacher_reply(sample_id):
    return {
        "id": "request-test-001",
        "usage": {"total_tokens": 123},
        "choices": [{"message": {"content": json.dumps({
            "schema_version": "teacher_output.v1",
            "sample_id": sample_id,
            "distribution": {"conflict_push": 1.0},
            "confidence": 1.0,
            "evidence": [{
                "type": "motion", "segment_id": "seg_001",
                "description": "measured interaction motion",
            }],
            "counter_evidence": [],
            "reason": "test response",
            "needs_review": False,
        })}}],
    }


def test_qwen_api_uses_environment_key_and_persists_no_secret(monkeypatch, tmp_path):
    captured = {}

    class Response:
        def read(self):
            return json.dumps(_teacher_reply("sample-a")).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

    def fake_open(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("TEST_DASHSCOPE_KEY", "top-secret-never-persist")
    monkeypatch.setattr(qwen_api.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(
        qwen_api, "summarize_pose_feature",
        lambda sample_id, feature: {"persons": [], "segments": [{"segment_id": "seg_001"}]},
    )
    monkeypatch.setattr(
        qwen_api, "_frame_data_urls",
        lambda video, count: (["data:image/jpeg;base64,AA=="] * count, {"decoder": "test"}),
    )
    config = {
        "enabled": True,
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-flash",
        "api_key_env": "TEST_DASHSCOPE_KEY",
        "timeout_seconds": 30,
        "max_attempts": 1,
        "max_frames": 2,
    }
    output = tmp_path / "teacher.json"
    value = qwen_api.run_qwen_api(
        config, sample_id="sample-a", feature=tmp_path / "feature.npz",
        pose_video=tmp_path / "pose.mp4", output=output,
    )

    assert value["result"]["label"] == "conflict_push"
    assert value["provenance"]["execution"] == "qwen_api_fallback"
    assert captured["authorization"] == "Bearer top-secret-never-persist"
    assert len(captured["payload"]["messages"][0]["content"]) == 3
    assert "top-secret-never-persist" not in output.read_text(encoding="utf-8")
    assert "top-secret-never-persist" not in json.dumps(value, ensure_ascii=False)


def test_qwen_api_requires_enabled_key_environment(monkeypatch):
    monkeypatch.delenv("MISSING_QWEN_KEY", raising=False)
    status = qwen_api.api_status({
        "enabled": True, "api_key_env": "MISSING_QWEN_KEY",
    })
    assert status["available"] is False
    assert status["key_present"] is False
    assert "MISSING_QWEN_KEY" in status["reason"]


def test_web_key_is_stored_separately_and_never_in_json_settings(tmp_path):
    settings_path = tmp_path / "qwen_api.json"
    secret_path = tmp_path / "qwen_api.secret"
    config = qwen_api.save_qwen_api(settings_path, {"enabled": True})
    qwen_api.save_qwen_api_secret(secret_path, "web-key-not-json")
    status = qwen_api.api_status(config, qwen_api.load_qwen_api_secret(secret_path))

    assert status["available"] is True
    assert status["credential_source"] == "web_settings"
    assert "web-key-not-json" not in settings_path.read_text(encoding="utf-8")
    assert qwen_api.load_qwen_api_secret(secret_path) == "web-key-not-json"
    qwen_api.clear_qwen_api_secret(secret_path)
    assert qwen_api.load_qwen_api_secret(secret_path) == ""


@pytest.mark.parametrize("endpoint", [
    "http://example.invalid/chat", "https://key@example.invalid/chat",
    "https://example.invalid/chat?api_key=secret",
])
def test_qwen_api_rejects_credentials_or_non_https_endpoint(endpoint):
    with pytest.raises(ValueError):
        qwen_api.normalize_qwen_api({"enabled": True, "endpoint": endpoint})


def test_teacher_falls_back_to_hosted_api_when_remote_teacher_fails(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        gpu_settings_path=tmp_path / "gpu.json", max_workers=1,
        repository_root=tmp_path, runtime_root=tmp_path / "runtime",
        qwen_remote=lambda: {"enabled": True},
        default_teacher_command=lambda: "",
        qwen_api_status=lambda: {"available": True},
        qwen_api=lambda: {"enabled": True},
    )
    manager = JobManager(settings, store=None)
    paths = {
        "feature": tmp_path / "feature.npz",
        "pose_video": tmp_path / "pose.mp4",
        "prediction": tmp_path / "prediction.json",
        "teacher": tmp_path / "teacher.json",
        "work_dir": tmp_path / "work",
    }
    paths["feature"].write_bytes(b"feature")
    paths["pose_video"].write_bytes(b"pose")
    called = []

    def failed_remote(*unused):
        raise RuntimeError("SSH unavailable")

    def hosted_api(config, **kwargs):
        called.append((config, kwargs))
        Path(kwargs["output"]).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("dahua_cup.backend.jobs.run_remote_qwen", failed_remote)
    monkeypatch.setattr("dahua_cup.backend.jobs.run_qwen_api", hosted_api)
    try:
        message = manager._execute_teacher("sample-b", tmp_path / "source.mp4", paths)
    finally:
        manager.shutdown()

    assert "Qwen API 备用教师" in message
    assert called and called[0][1]["sample_id"] == "sample-b"
