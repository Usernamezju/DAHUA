"""Runtime-adjustable macro parameters for the Campus6 review pipeline.

Values are persisted next to the GPU selection file under
``runtime/settings/macro_parameters.json`` and applied in-process on save, so
new jobs pick them up immediately.  Explicit environment variables still win
over both the saved file and the routing-config defaults.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# One entry per adjustable parameter.  "key" must match a Settings field.
MACRO_PARAMETERS = [
    {
        "key": "joint_score_threshold",
        "name": "关节点置信度阈值",
        "description": "RTMPose 提取时保留关节点的最低置信度，低于该值的关节点视为不可用",
        "type": "unit_interval",
        "default": 0.20,
        "group": "pose",
    },
    {
        "key": "teacher_trigger_confidence",
        "name": "难例门控置信度 τ",
        "description": "学生 Top-1 置信度 ≤ τ 时进入难例判定，高于 τ 跳过 Qwen",
        "type": "unit_interval",
        "default": 0.30,
        "group": "gate",
    },
    {
        "key": "teacher_trigger_margin",
        "name": "Top1-Top2 间隔阈值",
        "description": "Top-1 与 Top-2 概率间隔低于该值时判定为不确定",
        "type": "unit_interval",
        "default": 0.15,
        "group": "gate",
    },
    {
        "key": "pose_quality_threshold",
        "name": "骨架质量门控阈值",
        "description": "骨架覆盖与关节点分数加权质量低于该值视为质量不足",
        "type": "unit_interval",
        "default": 0.70,
        "group": "gate",
    },
    {
        "key": "teacher_conflict_confidence",
        "name": "师生冲突置信度阈值",
        "description": "学生与教师标签不同且双方置信度均达到该值时判为高置信冲突",
        "type": "unit_interval",
        "default": 0.70,
        "group": "gate",
    },
    {
        "key": "review_temperature",
        "name": "概率温度（审核软化）",
        "description": "审核界面六类概率的温度缩放系数，越大分布越平滑",
        "type": "positive_float",
        "default": 5.0,
        "group": "probability",
    },
    {
        "key": "priority_teacher_conflict",
        "name": "师生冲突复核优先级",
        "description": "高置信师生冲突样本进入复核队列的排序权重",
        "type": "nonnegative_int",
        "default": 1_000_000,
        "group": "priority",
    },
    {
        "key": "priority_pose_quality",
        "name": "骨架质量不足优先级",
        "description": "骨架质量不达标样本进入复核队列的排序权重",
        "type": "nonnegative_int",
        "default": 900_000,
        "group": "priority",
    },
    {
        "key": "priority_student_instability",
        "name": "学生不稳定优先级",
        "description": "时序不稳定样本进入复核队列的排序权重",
        "type": "nonnegative_int",
        "default": 800_000,
        "group": "priority",
    },
    {
        "key": "priority_teacher_review",
        "name": "教师要求复核优先级",
        "description": "教师主动要求人工复核样本的排序权重",
        "type": "nonnegative_int",
        "default": 750_000,
        "group": "priority",
    },
    {
        "key": "priority_teacher_unavailable",
        "name": "教师不可用优先级",
        "description": "学生不确定但教师未启用样本的排序权重",
        "type": "nonnegative_int",
        "default": 700_000,
        "group": "priority",
    },
]

# Environment variables that override each parameter.  A non-empty value
# counts as an explicit deployment-level override and takes precedence over
# both the saved file and the routing configuration.
ENV_OVERRIDES = {
    "joint_score_threshold": ("DAHUA_RTMPOSE_JOINT_SCORE_THRESHOLD",),
    "teacher_trigger_confidence": ("DAHUA_TEACHER_TRIGGER_CONFIDENCE",),
    "teacher_trigger_margin": ("DAHUA_TEACHER_TRIGGER_MARGIN",),
    "pose_quality_threshold": ("DAHUA_POSE_QUALITY_THRESHOLD",),
    "teacher_conflict_confidence": ("DAHUA_TEACHER_CONFLICT_CONFIDENCE",),
    "priority_teacher_conflict": ("DAHUA_PRIORITY_TEACHER_CONFLICT",),
    "priority_pose_quality": ("DAHUA_PRIORITY_POSE_QUALITY",),
    "priority_student_instability": ("DAHUA_PRIORITY_STUDENT_INSTABILITY",),
    "priority_teacher_review": ("DAHUA_PRIORITY_TEACHER_REVIEW",),
    "priority_teacher_unavailable": ("DAHUA_PRIORITY_TEACHER_UNAVAILABLE",),
    "review_temperature": (
        "DAHUA_CAMPUS6_REVIEW_TEMPERATURE",
        "DAHUA_CAMPUS6_PROBABILITY_TEMPERATURE",
    ),
}

SPEC_BY_KEY = {item["key"]: item for item in MACRO_PARAMETERS}
GROUP_NAMES = {
    "pose": "骨架提取",
    "gate": "难例门控",
    "probability": "概率校准",
    "priority": "复核优先级",
}


def _parse_value(key: str, raw: Any, *, strict: bool):
    """Coerce one raw value; invalid values raise when strict, else return None."""
    spec = SPEC_BY_KEY[key]
    kind = spec["type"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        if strict:
            raise ValueError("{} 必须是数字".format(spec["name"])) from None
        return None
    if not math.isfinite(value):
        if strict:
            raise ValueError("{} 必须是有限数字".format(spec["name"]))
        return None
    if kind == "unit_interval":
        if not 0 <= value <= 1:
            if strict:
                raise ValueError("{} 必须在 [0, 1] 区间内".format(spec["name"]))
            return None
        return float(value)
    if kind == "positive_float":
        if value <= 0:
            if strict:
                raise ValueError("{} 必须大于 0".format(spec["name"]))
            return None
        return float(value)
    # nonnegative_int
    if value != int(value) or value < 0:
        if strict:
            raise ValueError("{} 必须是非负整数".format(spec["name"]))
        return None
    return int(value)


def sanitize_values(values: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
    """Keep only known keys with well-formed values.

    Strict mode raises a ValueError describing the first invalid input;
    lenient mode silently drops invalid entries and is used when loading a
    possibly hand-edited settings file.
    """
    if not isinstance(values, dict):
        if strict:
            raise ValueError("宏参数必须是键值对象")
        return {}
    result = {}
    for key, raw in values.items():
        if key not in SPEC_BY_KEY:
            if strict:
                raise ValueError("未知宏参数：{}".format(key))
            continue
        parsed = _parse_value(key, raw, strict=strict)
        if parsed is not None:
            result[key] = parsed
    return result


def load_macro_parameter_file(path: Path) -> Dict[str, Any]:
    """Read persisted values, tolerating a missing or corrupted file."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return sanitize_values(value, strict=False)


def save_macro_parameter_file(path: Path, values: Dict[str, Any]) -> None:
    """Atomically write the sanitized values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _env_override_active(key: str) -> bool:
    return any(os.environ.get(name, "").strip() for name in ENV_OVERRIDES[key])


def snapshot(settings) -> dict:
    """Describe every macro parameter with its live value and provenance."""
    saved = load_macro_parameter_file(settings.macro_parameters_path)
    parameters = []
    for spec in MACRO_PARAMETERS:
        key = spec["key"]
        if _env_override_active(key):
            source = "environment"
        elif key in saved:
            source = "file"
        else:
            source = "default"
        parameters.append({
            "key": key,
            "name": spec["name"],
            "description": spec["description"],
            "type": spec["type"],
            "group": spec["group"],
            "default": spec["default"],
            "value": getattr(settings, key),
            "source": source,
        })
    return {
        "schema_version": "campus6_macro_parameters.v1",
        "parameters": parameters,
        "groups": GROUP_NAMES,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def apply_values(settings, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and set the values on the live Settings instance."""
    cleaned = sanitize_values(values, strict=True)
    for key, value in cleaned.items():
        setattr(settings, key, value)
    return cleaned
