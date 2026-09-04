import pytest
import sys
import threading
import time
from types import SimpleNamespace

from dahua_cup.backend.jobs import JobManager
from dahua_cup.backend.remote import (
    MMCV_WHEEL_INDEX,
    conda_layout,
    normalize_qwen_remote,
    provision_remote,
    run_remote_incremental_training,
    training_python_path,
)
from dahua_cup.backend.store import INTERRUPTED_JOB_MESSAGE


def test_remote_qwen_uses_only_ssh_and_project_root_fields():
    value = normalize_qwen_remote({
        "enabled": True,
        "host": "qwen.example.edu",
        "port": "2202",
        "user": "runner",
        "project_root": "/srv/DAHUA",
    })
    assert value == {
        "enabled": True,
        "host": "qwen.example.edu",
        "port": 2202,
        "user": "runner",
        "project_root": "/srv/DAHUA",
    }


@pytest.mark.parametrize("key,value", [
    ("host", "bad host"), ("user", "bad/user"), ("project_root", "relative/root"),
])
def test_enabled_remote_qwen_rejects_unsafe_connection_values(key, value):
    config = {
        "enabled": True, "host": "qwen.example.edu", "port": 22,
        "user": "runner", "project_root": "/srv/DAHUA",
    }
    config[key] = value
    with pytest.raises(ValueError):
        normalize_qwen_remote(config)


def test_shutdown_terminates_service_owned_worker_process(tmp_path):
    settings = SimpleNamespace(
        gpu_settings_path=tmp_path / "gpu.json", max_workers=1,
        repository_root=tmp_path,
    )
    manager = JobManager(settings, store=None)
    result = []

    def execute():
        try:
            manager._execute([sys.executable, "-c", "import time; time.sleep(30)"])
        except RuntimeError as exc:
            result.append(str(exc))

    thread = threading.Thread(target=execute)
    thread.start()
    deadline = time.monotonic() + 3
    while not manager._processes and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._processes
    manager.shutdown()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result == [INTERRUPTED_JOB_MESSAGE]


def test_incremental_training_uploads_only_frozen_annotation(monkeypatch, tmp_path):
    annotation = tmp_path / "training_annotations_with_all.pkl"
    annotation.write_bytes(b"frozen-pose-arrays")
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="remote training complete")

    monkeypatch.setattr("dahua_cup.backend.remote.subprocess.run", fake_run)
    result = run_remote_incremental_training(
        {
            "enabled": True,
            "host": "gpu.example.edu",
            "port": 2202,
            "user": "runner",
            "project_root": "/srv/DAHUA",
        },
        round_id="campus6_increment_001",
        annotation=annotation,
        remote_python="/opt/conda/envs/gcn/bin/python",
        init_checkpoint="/models/M0.fp32.pth",
        epochs=10,
    )
    assert result["remote_dir"].startswith("/srv/DAHUA/.runtime/incremental/")
    assert len(commands) == 3
    assert commands[1][0][-2] == str(annotation)
    assert "training_annotations_with_all.pkl" in commands[1][0][-1]
    remote_script = commands[2][0][-1]
    assert "train_campus6" in remote_script
    assert "--epochs 10" in remote_script
    assert "--distill" in remote_script
    assert "--lora" in remote_script
    assert "--deep-compression" not in remote_script
    assert "CUDA_VISIBLE_DEVICES" in remote_script


REMOTE_CONFIG = {
    "enabled": True, "host": "gpu.example.edu", "port": 2202,
    "user": "runner", "project_root": "/srv/DAHUA",
}


def test_conda_layout_splits_standard_conda_env_path():
    assert conda_layout("/root/miniconda3/envs/skel_gcn38/bin/python") == (
        "/root/miniconda3", "skel_gcn38")
    assert conda_layout("/opt/conda/envs/gcn/bin/python3.8") == (
        "/opt/conda", "gcn")


@pytest.mark.parametrize("path", [
    "/opt/venv/bin/python",
    "/root/miniconda3/envs/skel_gcn38",
    "relative/envs/gcn/bin/python",
])
def test_conda_layout_rejects_non_conda_paths(path):
    assert conda_layout(path) is None


def test_training_python_path_prefers_environment(monkeypatch):
    monkeypatch.delenv("DAHUA_INCREMENTAL_REMOTE_PYTHON", raising=False)
    assert training_python_path() == "/root/miniconda3/envs/skel_gcn38/bin/python"
    monkeypatch.setenv("DAHUA_INCREMENTAL_REMOTE_PYTHON", "/opt/conda/envs/gcn/bin/python")
    assert training_python_path() == "/opt/conda/envs/gcn/bin/python"


def test_provision_remote_skips_existing_training_environment(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if command[-1].startswith("test -x"):
            return SimpleNamespace(stdout="", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr("dahua_cup.backend.remote.subprocess.run", fake_run)
    summary = provision_remote(dict(REMOTE_CONFIG))
    assert summary == {"qwen_env": "ready", "training_env": "already_present"}
    # Only the checkout/venv script and the existence probe run; no build.
    assert len(commands) == 2
    assert commands[1][0][-1] == "test -x /root/miniconda3/envs/skel_gcn38/bin/python"
    assert commands[1][1]["timeout"] == 60


def test_provision_remote_creates_missing_training_environment(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if command[-1].startswith("test -x"):
            return SimpleNamespace(stdout="", returncode=1)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr("dahua_cup.backend.remote.subprocess.run", fake_run)
    summary = provision_remote(dict(REMOTE_CONFIG))
    assert summary == {"qwen_env": "ready", "training_env": "created"}
    assert len(commands) == 3
    build = commands[2][0][-1]
    assert build.startswith("sh -lc ")
    assert "create -n skel_gcn38 python=3.8" in build
    assert "pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.3" in build
    assert "/srv/DAHUA/requirements/skel.txt" in build
    assert MMCV_WHEEL_INDEX in build
    assert commands[2][1]["timeout"] == 3600


def test_provision_remote_rejects_uncreatable_training_python(monkeypatch):
    def fake_run(command, **kwargs):
        if command[-1].startswith("test -x"):
            return SimpleNamespace(stdout="", returncode=1)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr("dahua_cup.backend.remote.subprocess.run", fake_run)
    monkeypatch.setenv("DAHUA_INCREMENTAL_REMOTE_PYTHON", "/opt/venv/bin/python")
    with pytest.raises(ValueError, match="Conda"):
        provision_remote(dict(REMOTE_CONFIG))


def test_provision_remote_raises_when_training_probe_fails(monkeypatch):
    def fake_run(command, **kwargs):
        if command[-1].startswith("test -x"):
            return SimpleNamespace(stdout="ssh: connect failed", returncode=255)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr("dahua_cup.backend.remote.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="检查云端训练环境失败"):
        provision_remote(dict(REMOTE_CONFIG))
