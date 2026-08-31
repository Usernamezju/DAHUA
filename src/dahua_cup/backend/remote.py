"""SSH transport for a Qwen teacher hosted in another DAHUA checkout."""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


DEFAULT_QWEN_REMOTE = {
    "enabled": False,
    "host": "",
    "port": 22,
    "user": "",
    "project_root": "",
}

_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,252}$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_qwen_remote(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the small, deployment-neutral SSH configuration surface."""
    if not isinstance(value, dict):
        raise ValueError("Qwen 连接设置必须是对象")
    result = {**DEFAULT_QWEN_REMOTE, **value}
    result["enabled"] = bool(result["enabled"])
    result["host"] = str(result["host"] or "").strip()
    result["user"] = str(result["user"] or "").strip()
    result["project_root"] = str(result["project_root"] or "").strip()
    try:
        result["port"] = int(result["port"])
    except (TypeError, ValueError):
        raise ValueError("Qwen SSH 端口必须是整数") from None
    if not 1 <= result["port"] <= 65535:
        raise ValueError("Qwen SSH 端口必须在 1 到 65535 之间")
    if result["enabled"]:
        if not _HOST.fullmatch(result["host"]):
            raise ValueError("Qwen 主机名或 IP 地址无效")
        if not _USER.fullmatch(result["user"]):
            raise ValueError("Qwen SSH 用户名无效")
        if not result["project_root"].startswith("/") or "\x00" in result["project_root"]:
            raise ValueError("云端项目根目录必须是绝对路径")
    return result


def load_qwen_remote(path: Path) -> dict[str, Any]:
    try:
        return normalize_qwen_remote(_read_json(path))
    except ValueError:
        return dict(DEFAULT_QWEN_REMOTE)


def save_qwen_remote(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    cleaned = normalize_qwen_remote(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return cleaned


def _target(config: dict[str, Any]) -> str:
    return "{}@{}".format(config["user"], config["host"])


def test_connection(config: dict[str, Any]) -> None:
    if not config["enabled"]:
        raise ValueError("请先启用 Qwen 云端连接")
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", str(config["port"]), _target(config), "test -d " + shlex.quote(config["project_root"])],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=12,
    )


def provision_remote(config: dict[str, Any]) -> None:
    """Create/update the remote checkout and its project-local Qwen venv."""
    if not config["enabled"]:
        raise ValueError("请先启用 Qwen 云端连接")
    root = config["project_root"].rstrip("/")
    repository = os.environ.get(
        "DAHUA_REMOTE_REPOSITORY_URL", "https://github.com/Usernamezju/DAHUA.git"
    )
    script = """set -eu
if test -d {root}/.git; then
  cd {root}
  git pull --ff-only
else
  mkdir -p {parent}
  git clone {repository} {root}
  cd {root}
fi
if ! test -x .venv-qwen/bin/python; then
  python3 -m venv .venv-qwen
fi
.venv-qwen/bin/python -m pip install --upgrade pip
.venv-qwen/bin/python -m pip install -r requirements/server.txt
""".format(
        root=shlex.quote(root), parent=shlex.quote(posixpath.dirname(root)),
        repository=shlex.quote(repository),
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-p", str(config["port"]), _target(config), "sh -lc " + shlex.quote(script)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1800,
    )


def run_remote_qwen(
    config: dict[str, Any], sample_id: str, paths: dict[str, Path],
    execute: Optional[Callable[[list[str]], str]] = None,
) -> None:
    """Run the worker relative to the remote root using its own environment."""
    if not config["enabled"]:
        raise ValueError("Qwen 云端连接未启用")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    if not safe:
        raise ValueError("无效的样本 ID")
    root = config["project_root"].rstrip("/")
    relative_dir = ".runtime/qwen-remote/{}-{}".format(safe, uuid.uuid4().hex)
    remote_dir = root + "/" + relative_dir
    target = _target(config)
    ssh = ["ssh", "-o", "BatchMode=yes", "-p", str(config["port"]), target]
    scp = ["scp", "-o", "BatchMode=yes", "-P", str(config["port"])]

    def remote(script: str, *, timeout: int = 900) -> None:
        if execute is not None:
            execute(ssh + [script])
            return
        subprocess.run(ssh + [script], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=timeout)

    try:
        remote("mkdir -p " + shlex.quote(remote_dir))
        uploads = [paths["feature"], paths["pose_video"], paths["prediction"]]
        upload = scp + [*(str(path) for path in uploads), target + ":" + remote_dir + "/"]
        if execute is not None:
            execute(upload)
        else:
            subprocess.run(upload, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=300)
        command = [
            ".venv-qwen/bin/python", "-m", "dahua_cup.pipeline.qwen_teacher_worker",
            "--sample-id", sample_id,
            "--feature", relative_dir + "/" + paths["feature"].name,
            "--pose-video", relative_dir + "/" + paths["pose_video"].name,
            "--student-json", relative_dir + "/" + paths["prediction"].name,
            "--output", relative_dir + "/teacher.json",
        ]
        remote("cd {} && PYTHONPATH=src${{PYTHONPATH:+:$PYTHONPATH}} {}".format(
            shlex.quote(root), shlex.join(command)
        ))
        paths["teacher"].parent.mkdir(parents=True, exist_ok=True)
        download = scp + [target + ":" + remote_dir + "/teacher.json", str(paths["teacher"])]
        if execute is not None:
            execute(download)
        else:
            subprocess.run(download, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)
    finally:
        try:
            remote("rm -rf " + shlex.quote(remote_dir), timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
