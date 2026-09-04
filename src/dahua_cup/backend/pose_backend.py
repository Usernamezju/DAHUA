"""Persistent selection and readiness checks for pose-extraction backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_POSE_BACKEND = "tensorrt_fp16"
POSE_BACKENDS = {
    "mmpose_fp32": {
        "name": "RTMDet-S + RTMPose-S（FP32）",
        "description": "兼容/诊断用 MMPose 路径；COCO-17 输出兼容既有 ProtoGCN 数据。",
        "runtime": "MMPose",
    },
    "tensorrt_fp16": {
        "name": "RTMDet-nano-person + RTMPose-S（TensorRT FP16）",
        "description": "默认部署路径：TensorRT FP16 引擎；仅影响后续骨架提取，不改写既有骨架文件。",
        "runtime": "MMDeploy / TensorRT",
    },
    "tensorrt_int8": {
        "name": "RTMDet-S + RTMPose-S（TensorRT INT8）",
        "description": "默认部署路径：经校准、验证的 TensorRT INT8 引擎；仅影响后续骨架提取。",
        "runtime": "MMDeploy / TensorRT",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _engine_set_ready(root: Path, detector_name: str) -> bool:
    required = (
        root / detector_name / "deploy.json",
        root / detector_name / "end2end.engine",
        root / "rtmpose_s" / "deploy.json",
        root / "rtmpose_s" / "end2end.engine",
    )
    if not all(path.is_file() for path in required):
        return False
    validation = _read_json(root / "validation.json")
    return (
        validation.get("status") == "passed"
        and validation.get("detector", {}).get("status") == "passed"
        and validation.get("pose", {}).get("status") == "passed"
    )


def _int8_ready(root: Path) -> bool:
    return _engine_set_ready(root, "rtmdet_s")


def _fp16_ready(root: Path) -> bool:
    return _engine_set_ready(root, "rtmdet_nano_person")


def load_pose_backend(path: Path) -> str:
    configured = os.environ.get("DAHUA_POSE_BACKEND", "").strip().lower()
    if configured:
        if configured not in POSE_BACKENDS:
            raise ValueError(
                "DAHUA_POSE_BACKEND must be mmpose_fp32, tensorrt_fp16 or tensorrt_int8"
            )
        return configured
    selected = str(_read_json(path).get("selected") or DEFAULT_POSE_BACKEND)
    return selected if selected in POSE_BACKENDS else DEFAULT_POSE_BACKEND


def pose_backend_status(
    path: Path,
    int8_root: Path,
    *,
    fp32_available: bool,
    fp16_root: Optional[Path] = None,
) -> dict:
    selected = load_pose_backend(path)
    fp16_root = fp16_root or int8_root.parent / "fp16"
    choices = []
    for backend_id, definition in POSE_BACKENDS.items():
        available = (
            fp32_available
            if backend_id == "mmpose_fp32"
            else _fp16_ready(fp16_root)
            if backend_id == "tensorrt_fp16"
            else _int8_ready(int8_root)
        )
        choices.append({
            "id": backend_id,
            **definition,
            "available": available,
            "selected": backend_id == selected,
        })
    selected_available = next(
        item["available"] for item in choices if item["id"] == selected
    )
    return {
        "schema_version": "campus6_pose_backend.v1",
        "selected": selected,
        "selected_available": selected_available,
        "fp16_model_root": str(fp16_root),
        "int8_model_root": str(int8_root),
        "choices": choices,
    }


def save_pose_backend(
    path: Path,
    backend_id: str,
    int8_root: Path,
    *,
    fp32_available: bool,
    fp16_root: Optional[Path] = None,
) -> dict:
    if backend_id not in POSE_BACKENDS:
        raise ValueError("未知的骨架提取模型")
    status = pose_backend_status(
        path, int8_root, fp32_available=fp32_available, fp16_root=fp16_root
    )
    choice = next(item for item in status["choices"] if item["id"] == backend_id)
    if not choice["available"]:
        raise ValueError("所选骨架提取模型尚不可用；请先完成引擎导出与验证")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"selected": backend_id}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return pose_backend_status(
        path, int8_root, fp32_available=fp32_available, fp16_root=fp16_root
    )
