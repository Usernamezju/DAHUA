import pytest
import sys
import threading
import time
from types import SimpleNamespace

from dahua_cup.backend.jobs import JobManager
from dahua_cup.backend.remote import normalize_qwen_remote
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
