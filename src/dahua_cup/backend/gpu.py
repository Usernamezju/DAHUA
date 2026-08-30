"""GPU discovery, persisted device pools, and round-robin assignment."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, List


def _unique_gpu_ids(values) -> List[int]:
    result = []
    for value in values:
        gpu_id = int(value)
        if gpu_id < 0:
            raise ValueError("GPU 编号不能为负数")
        if gpu_id not in result:
            result.append(gpu_id)
    return result


def parse_nvidia_smi(output: str) -> List[dict]:
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            gpus.append({
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mb": int(fields[3]),
                "memory_used_mb": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
                "temperature_c": int(fields[6]),
            })
        except ValueError:
            continue
    return gpus


class GPUManager:
    """Persist the selectable ProtoGCN GPU pool; MediaPipe runs on CPU."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cursor: Dict[str, int] = {"pose": 0, "student": 0}
        self._selection = self._load()

    @staticmethod
    def _env_defaults() -> dict:
        shared = os.environ.get("DAHUA_INFERENCE_GPUS", "0")
        student = os.environ.get("DAHUA_PROTOGCN_GPUS", shared)

        def parse(value: str) -> List[int]:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return _unique_gpu_ids(parts or [0])

        return {
            "pose_gpu_ids": [],
            "student_gpu_ids": parse(student),
        }

    def _load(self) -> dict:
        defaults = self._env_defaults()
        if not self.path.is_file():
            return defaults
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return {
                "pose_gpu_ids": [],
                "student_gpu_ids": _unique_gpu_ids(
                    value.get("student_gpu_ids", defaults["student_gpu_ids"])
                ),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return defaults

    def discover(self) -> List[dict]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        return parse_nvidia_smi(completed.stdout)

    def state(self) -> dict:
        gpus = self.discover()
        available_ids = {gpu["index"] for gpu in gpus}
        return {
            "available": bool(gpus),
            "gpus": gpus,
            "pose_gpu_ids": [],
            "student_gpu_ids": list(self._selection["student_gpu_ids"]),
            "unavailable_pose_gpu_ids": [],
            "unavailable_student_gpu_ids": sorted(
                set(self._selection["student_gpu_ids"]) - available_ids
            ),
            "strategy": "round_robin",
            "mediapipe_device": "cpu",
            "mediapipe_delegate": "CPU",
        }

    def update(self, pose_gpu_ids, student_gpu_ids) -> dict:
        pose = []
        student = _unique_gpu_ids(student_gpu_ids)
        if not student:
            raise ValueError("ProtoGCN 至少需要选择一张 GPU")
        available = {gpu["index"] for gpu in self.discover()}
        if available:
            invalid = (set(pose) | set(student)) - available
            if invalid:
                raise ValueError("GPU 不存在或当前不可见：{}".format(
                    ", ".join(str(value) for value in sorted(invalid))
                ))
        selection = {"pose_gpu_ids": pose, "student_gpu_ids": student}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        with self._lock:
            self._selection = selection
            self._cursor = {"pose": 0, "student": 0}
        return self.state()

    def next_device(self, kind: str) -> int:
        if kind not in self._cursor:
            raise ValueError("unknown GPU workload: {}".format(kind))
        key = "{}_gpu_ids".format(kind)
        with self._lock:
            pool = self._selection[key]
            if not pool:
                raise RuntimeError("{} GPU 池为空".format(kind))
            index = self._cursor[kind] % len(pool)
            self._cursor[kind] += 1
            return pool[index]

    @staticmethod
    def process_environment(gpu_id: int) -> dict:
        """Expose one physical GPU; child processes address it as logical GPU 0."""
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "DAHUA_PHYSICAL_GPU_ID": str(gpu_id),
        }
