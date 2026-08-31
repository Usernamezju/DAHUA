import pytest

from dahua_cup.backend.remote import normalize_qwen_remote


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
