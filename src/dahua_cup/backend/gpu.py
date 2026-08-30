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
    """Select the least-loaded permitted GPU for pose and Campus6 jobs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._leases: Dict[str, Dict[int, int]] = {"pose": {}, "student": {}}
        self._teacher_active_ids: List[int] = []
        self._selection = self._load()

    @staticmethod
    def _env_defaults() -> dict:
        shared = os.environ.get("DAHUA_INFERENCE_GPUS", "0")
        pose = os.environ.get("DAHUA_RTMPOSE_GPUS", shared)
        student = os.environ.get("DAHUA_PROTOGCN_GPUS", shared)
        teacher = os.environ.get("DAHUA_QWEN_GPUS", "")

        def parse(value: str) -> List[int]:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return _unique_gpu_ids(parts or [0])

        return {
            "pose_gpu_ids": parse(pose),
            "student_gpu_ids": parse(student),
            "teacher_mode": "manual" if teacher.strip() else "auto",
            "teacher_gpu_ids": parse(teacher) if teacher.strip() else [],
        }

    def _load(self) -> dict:
        defaults = self._env_defaults()
        if not self.path.is_file():
            return defaults
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            pose = _unique_gpu_ids(
                value.get("pose_gpu_ids", defaults["pose_gpu_ids"])
            )
            student = _unique_gpu_ids(
                value.get("student_gpu_ids", defaults["student_gpu_ids"])
            )
            teacher_mode = str(
                value.get("teacher_mode", defaults["teacher_mode"])
            )
            teacher = _unique_gpu_ids(
                value.get("teacher_gpu_ids", defaults["teacher_gpu_ids"])
            )
            if teacher_mode not in {"auto", "manual"}:
                teacher_mode = "auto"
            return {
                # Upgrade the old CPU-pose settings file without requiring an
                # operator to open the settings page first.
                "pose_gpu_ids": pose or defaults["pose_gpu_ids"],
                "student_gpu_ids": student or defaults["student_gpu_ids"],
                "teacher_mode": teacher_mode,
                "teacher_gpu_ids": teacher,
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
        with self._lock:
            pose_leases = dict(self._leases["pose"])
            student_leases = dict(self._leases["student"])
            teacher_active = set(self._teacher_active_ids)
        for gpu in gpus:
            gpu_id = gpu["index"]
            roles = []
            if pose_leases.get(gpu_id, 0):
                roles.append("RTMDet/RTMPose")
            if student_leases.get(gpu_id, 0):
                roles.append("M1KD")
            if gpu_id in teacher_active:
                roles.append("Qwen3-VL-8B")
            gpu["service_roles"] = roles
            gpu["service_busy"] = bool(roles)
        return {
            "available": bool(gpus),
            "gpus": gpus,
            "pose_gpu_ids": list(self._selection["pose_gpu_ids"]),
            "student_gpu_ids": list(self._selection["student_gpu_ids"]),
            "teacher_mode": self._selection["teacher_mode"],
            "teacher_gpu_ids": list(self._selection["teacher_gpu_ids"]),
            "unavailable_pose_gpu_ids": sorted(
                set(self._selection["pose_gpu_ids"]) - available_ids
            ),
            "unavailable_student_gpu_ids": sorted(
                set(self._selection["student_gpu_ids"]) - available_ids
            ),
            "unavailable_teacher_gpu_ids": sorted(
                set(self._selection["teacher_gpu_ids"]) - available_ids
            ),
            "teacher_admission": self.teacher_availability(),
            "strategy": "least_loaded_then_lowest_utilization",
            "pose_runtime": "RTMDet/RTMPose on CUDA",
        }

    def teacher_availability(self) -> dict:
        """Return GPUs that are genuinely idle enough for the server Qwen model.

        Qwen3-VL-8B is loaded from the server's shared model volume in fp16.
        It needs one almost-empty 24 GiB card.  The admission check remains
        conservative so an on-demand teacher never evicts another user's job.
        """
        required = max(1, int(os.environ.get("DAHUA_QWEN_MIN_GPUS", "1")))
        minimum_free = max(
            1, int(os.environ.get("DAHUA_QWEN_MIN_FREE_MEMORY_MB", "20000"))
        )
        maximum_utilization = max(
            0, min(100, int(os.environ.get("DAHUA_QWEN_MAX_UTILIZATION", "10")))
        )
        permitted = set(self._selection["teacher_gpu_ids"])
        automatic = self._selection["teacher_mode"] == "auto"
        idle = [
            gpu for gpu in self.discover()
            if (automatic or gpu["index"] in permitted)
            if gpu["memory_total_mb"] - gpu["memory_used_mb"] >= minimum_free
            and gpu["utilization_gpu_percent"] <= maximum_utilization
        ]
        idle.sort(key=lambda gpu: (
            gpu["utilization_gpu_percent"],
            gpu["memory_used_mb"],
            gpu["index"],
        ))
        selected = [gpu["index"] for gpu in idle[:required]]
        return {
            "enabled": len(selected) == required,
            "gpu_ids": selected,
            "required_gpu_count": required,
            "minimum_free_memory_mb": minimum_free,
            "maximum_utilization_percent": maximum_utilization,
            "idle_gpu_ids": [gpu["index"] for gpu in idle],
            "selection_mode": self._selection["teacher_mode"],
            "permitted_gpu_ids": (
                sorted(permitted) if not automatic else []
            ),
        }

    @staticmethod
    def teacher_process_environment(gpu_ids: List[int]) -> dict:
        if not gpu_ids:
            raise ValueError("Qwen requires at least one selected GPU")
        visible = ",".join(str(value) for value in gpu_ids)
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": visible,
            "DAHUA_QWEN_GPU_IDS": visible,
        }

    def update(
        self,
        pose_gpu_ids,
        student_gpu_ids,
        teacher_gpu_ids=None,
        *,
        teacher_auto: bool = True,
    ) -> dict:
        pose = _unique_gpu_ids(pose_gpu_ids)
        student = _unique_gpu_ids(student_gpu_ids)
        teacher = _unique_gpu_ids(teacher_gpu_ids or [])
        if not pose or not student:
            raise ValueError("RTMPose 和 ProtoGCN 都至少需要选择一张 GPU")
        if not teacher_auto and not teacher:
            raise ValueError("手动模式下，Qwen 至少需要选择一张候选 GPU")
        available = {gpu["index"] for gpu in self.discover()}
        if available:
            invalid = (set(pose) | set(student) | set(teacher)) - available
            if invalid:
                raise ValueError("GPU 不存在或当前不可见：{}".format(
                    ", ".join(str(value) for value in sorted(invalid))
                ))
        selection = {
            "pose_gpu_ids": pose,
            "student_gpu_ids": student,
            "teacher_mode": "auto" if teacher_auto else "manual",
            "teacher_gpu_ids": teacher,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        with self._lock:
            self._selection = selection
            self._leases = {"pose": {}, "student": {}}
        return self.state()

    def set_teacher_active(self, gpu_ids) -> None:
        """Expose the service-owned Qwen allocation in the monitoring API."""
        with self._lock:
            self._teacher_active_ids = _unique_gpu_ids(gpu_ids)

    def acquire_device(self, kind: str) -> int:
        """Lease the currently best GPU in the selected pool.

        Ranking first avoids cards already leased by this service, then uses
        live nvidia-smi utilization and used-memory ratio.  It deliberately
        never guesses a CPU fallback: unavailable CUDA capacity is a visible
        job submission error.
        """
        if kind not in self._leases:
            raise ValueError("unknown GPU workload: {}".format(kind))
        key = "{}_gpu_ids".format(kind)
        with self._lock:
            pool = self._selection[key]
            if not pool:
                raise RuntimeError("{} GPU 池为空".format(kind))
            snapshots = {gpu["index"]: gpu for gpu in self.discover()}
            available = [gpu_id for gpu_id in pool if gpu_id in snapshots]
            if not available:
                raise RuntimeError("{} GPU 池中的设备当前均不可见".format(kind))

            def rank(gpu_id: int):
                gpu = snapshots[gpu_id]
                total = max(gpu["memory_total_mb"], 1)
                return (
                    self._leases[kind].get(gpu_id, 0),
                    gpu["utilization_gpu_percent"],
                    gpu["memory_used_mb"] / total,
                    gpu["memory_used_mb"],
                    gpu_id,
                )

            selected = min(available, key=rank)
            self._leases[kind][selected] = self._leases[kind].get(selected, 0) + 1
            return selected

    def release_device(self, kind: str, gpu_id: int) -> None:
        if kind not in self._leases:
            return
        with self._lock:
            active = self._leases[kind].get(gpu_id, 0)
            if active <= 1:
                self._leases[kind].pop(gpu_id, None)
            else:
                self._leases[kind][gpu_id] = active - 1

    @staticmethod
    def process_environment(gpu_id: int) -> dict:
        """Expose one physical GPU; child processes address it as logical GPU 0."""
        return {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "DAHUA_PHYSICAL_GPU_ID": str(gpu_id),
        }
