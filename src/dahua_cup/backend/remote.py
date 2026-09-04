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

# Server-side GCN training environment.  Mirrors the default in
# Settings.incremental_remote_python; the environment variable is the single
# source of truth for which remote Python incremental training must use.
DEFAULT_REMOTE_TRAINING_PYTHON = "/root/miniconda3/envs/skel_gcn38/bin/python"
MMCV_WHEEL_INDEX = "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"

_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,252}$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_CONDA_ENV_PYTHON = re.compile(r"^(.+)/envs/([A-Za-z0-9_.-]+)/bin/python[0-9.]*$")


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


def training_python_path() -> str:
    """Return the remote Python used for server-side GCN training."""
    return os.environ.get(
        "DAHUA_INCREMENTAL_REMOTE_PYTHON", DEFAULT_REMOTE_TRAINING_PYTHON
    ).strip()


def conda_layout(python_path: str) -> Optional[tuple[str, str]]:
    """Split ``<conda-root>/envs/<name>/bin/python`` into (conda root, name).

    Returns ``None`` when the path cannot host an auto-created Conda env.
    """
    value = python_path.rstrip("/")
    if not value.startswith("/"):
        return None
    match = _CONDA_ENV_PYTHON.fullmatch(value)
    if not match:
        return None
    return match.group(1), match.group(2)


def test_connection(config: dict[str, Any]) -> None:
    if not config["enabled"]:
        raise ValueError("请先启用 Qwen 云端连接")
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", str(config["port"]), _target(config), "test -d " + shlex.quote(config["project_root"])],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=12,
    )


def provision_remote(config: dict[str, Any]) -> dict[str, str]:
    """Create/update the remote checkout, its Qwen venv and GCN training env."""
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
    return {"qwen_env": "ready", "training_env": _provision_training_env(config)}


def _provision_training_env(config: dict[str, Any]) -> str:
    """Ensure the remote GCN Conda environment used by incremental training.

    The Python path is deployment state shared with
    ``Settings.incremental_remote_python``; an existing interpreter is left
    untouched (the competition server already ships ``skel_gcn38``).  A
    missing environment is recreated from ``requirements/skel.txt``: Python
    3.8 with the PyTorch 1.10.2 / CUDA 11.3 Conda build, then the pinned pip
    set including the prebuilt mmcv-full 1.5.0 wheel.
    """
    python = training_python_path()
    if not python:
        raise ValueError("DAHUA_INCREMENTAL_REMOTE_PYTHON 不能为空")
    ssh = ["ssh", "-o", "BatchMode=yes", "-p", str(config["port"]), _target(config)]
    probe = subprocess.run(
        ssh + ["test -x " + shlex.quote(python)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
    )
    if probe.returncode == 0:
        return "already_present"
    if probe.returncode != 1:
        raise RuntimeError(
            "检查云端训练环境失败：{}".format((probe.stdout or "").strip()[:300])
        )
    layout = conda_layout(python)
    if layout is None:
        raise ValueError(
            "云端训练 Python 不存在，且路径不符合 Conda 环境布局"
            "（<conda 根>/envs/<环境名>/bin/python），无法自动创建：{}".format(python)
        )
    conda_root, env_name = layout
    conda_bin = conda_root + "/bin/conda"
    skel = config["project_root"].rstrip("/") + "/requirements/skel.txt"
    script = """set -eu
conda_bin="$(command -v conda || true)"
test -n "$conda_bin" || conda_bin={conda}
test -x "$conda_bin" || {{ echo 'Conda is unavailable on the remote host' >&2; exit 2; }}
test -f {skel} || {{ echo 'requirements/skel.txt is missing; push it and re-run' >&2; exit 2; }}
"$conda_bin" create -n {env} python=3.8 -y
"$conda_bin" install -n {env} -y -c pytorch pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.3
{py} -m pip install --upgrade pip
{py} -m pip install -r {skel} -f {mmcv_index}
""".format(
        conda=shlex.quote(conda_bin),
        skel=shlex.quote(skel),
        env=shlex.quote(env_name),
        py=shlex.quote(python),
        mmcv_index=shlex.quote(MMCV_WHEEL_INDEX),
    )
    subprocess.run(
        ssh + ["sh -lc " + shlex.quote(script)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=3600,
    )
    return "created"


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


def run_remote_incremental_training(
    config: dict[str, Any],
    *,
    round_id: str,
    annotation: Path,
    remote_python: str,
    init_checkpoint: str,
    epochs: int,
    gpu_id: str = "auto",
) -> dict[str, str]:
    """Upload a frozen annotation and run Campus6 fine-tuning remotely.

    The pickle contains the COCO-17 arrays, so raw videos never leave the
    workstation.  Output remains on the server for checkpoint provenance; the
    Upload acknowledgement is not a successful training result: callers must
    retain their staging transaction until this function returns successfully.
    """
    if not config.get("enabled"):
        raise ValueError("增量训练服务器未配置；请先启用 Qwen SSH 连接")
    annotation = Path(annotation)
    if not annotation.is_file():
        raise FileNotFoundError("incremental training annotation is missing")
    if not remote_python.startswith("/") or not init_checkpoint.startswith("/"):
        raise ValueError("远程 Python 与初始化权重必须使用绝对路径")
    if epochs < 1:
        raise ValueError("incremental training epochs must be positive")
    if gpu_id != "auto" and not gpu_id.isdigit():
        raise ValueError("DAHUA_INCREMENTAL_REMOTE_GPU_ID must be auto or a GPU index")

    safe_round = re.sub(r"[^A-Za-z0-9_.-]+", "_", round_id).strip("._")
    if not safe_round:
        raise ValueError("invalid incremental round id")
    root = str(config["project_root"]).rstrip("/")
    relative = ".runtime/incremental/{}-{}".format(safe_round, uuid.uuid4().hex)
    remote_dir = root + "/" + relative
    target = _target(config)
    ssh = ["ssh", "-o", "BatchMode=yes", "-p", str(config["port"]), target]
    scp = ["scp", "-o", "BatchMode=yes", "-P", str(config["port"])]

    def invoke(command: list[str], timeout: int) -> str:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
        return completed.stdout or ""

    invoke(ssh + ["mkdir -p " + shlex.quote(remote_dir)], timeout=30)
    invoke(
        scp + [
            str(annotation),
            target + ":" + remote_dir + "/training_annotations_with_all.pkl",
        ],
        timeout=300,
    )
    upload = {
        "remote_dir": remote_dir,
        "remote_annotation": remote_dir + "/training_annotations_with_all.pkl",
    }
    if gpu_id == "auto":
        gpu_script = (
            "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu "
            "--format=csv,noheader,nounits | awk -F, '$2-$3 >= 20000 && $4 <= 10 "
            "{gsub(/ /, \"\", $1); print $1; exit}'"
        )
    else:
        gpu_script = "printf '%s\\n' " + shlex.quote(gpu_id)
    command = """set -eu
gpu_id=$({gpu_script})
test -n \"$gpu_id\" || {{ echo 'No idle GPU available for incremental training' >&2; exit 75; }}
test -x {python} || {{ echo 'Remote GCN Python is unavailable' >&2; exit 2; }}
test -f {checkpoint} || {{ echo 'Remote FP32 initialization checkpoint is unavailable' >&2; exit 2; }}
cd {root}
CUDA_VISIBLE_DEVICES=\"$gpu_id\" PYTHONPATH=src${{PYTHONPATH:+:$PYTHONPATH}} {python} -m dahua_cup.pipeline.train_campus6 \\
  --ann-file {annotation} --work-dir {work_dir} --init-checkpoint {checkpoint} \\
  --epochs {epochs} --gpus 1 --distill --lora
""".format(
        gpu_script=gpu_script,
        python=shlex.quote(remote_python),
        checkpoint=shlex.quote(init_checkpoint),
        root=shlex.quote(root),
        annotation=shlex.quote(relative + "/training_annotations_with_all.pkl"),
        work_dir=shlex.quote(relative + "/work_dir"),
        epochs=int(epochs),
    )
    output = invoke(ssh + ["sh -lc " + shlex.quote(command)], timeout=24 * 3600)
    return {
        **upload,
        "log_tail": output[-4000:],
    }
