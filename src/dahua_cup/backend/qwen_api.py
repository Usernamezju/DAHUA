"""OpenAI-compatible hosted Qwen fallback for the Campus6 teacher.

Only non-secret configuration is persisted.  The bearer token is always read
from the named environment variable when a request is made.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dahua_cup.feature_extraction.semantic_graph import summarize_pose_feature
from dahua_cup.semantic_teacher.api.qwen3_backend import (
    extract_json_object,
    parse_teacher_output,
)
from dahua_cup.semantic_teacher.prompts.prompt_builder import (
    PROMPT_VERSION,
    build_teacher_prompt,
)
from dahua_cup.semantic_teacher.schemas import LABELS


DEFAULT_QWEN_API = {
    "enabled": False,
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "model": "qwen3-vl-flash",
    "api_key_env": "DASHSCOPE_API_KEY",
    "timeout_seconds": 180,
    "max_attempts": 2,
    "max_frames": 8,
}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MODEL = re.compile(r"^[A-Za-z0-9_.:/-]{1,160}$")


def normalize_qwen_api(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Qwen API 设置必须是对象")
    result = {**DEFAULT_QWEN_API, **value}
    result["enabled"] = bool(result["enabled"])
    result["endpoint"] = str(result["endpoint"] or "").strip()
    result["model"] = str(result["model"] or "").strip()
    result["api_key_env"] = str(result["api_key_env"] or "").strip()
    try:
        result["timeout_seconds"] = int(result["timeout_seconds"])
        result["max_attempts"] = int(result["max_attempts"])
        result["max_frames"] = int(result["max_frames"])
    except (TypeError, ValueError):
        raise ValueError("Qwen API 超时、重试次数和帧数必须是整数") from None
    endpoint = urllib.parse.urlparse(result["endpoint"])
    if result["endpoint"] and (
        endpoint.scheme != "https" or not endpoint.netloc
        or endpoint.username or endpoint.password
        or endpoint.query or endpoint.fragment
    ):
        raise ValueError(
            "Qwen API Endpoint 必须是无凭据、无查询参数的 HTTPS 地址"
        )
    if result["enabled"]:
        if not result["endpoint"]:
            raise ValueError("启用 Qwen API 时必须填写 Endpoint")
        if not _MODEL.fullmatch(result["model"]):
            raise ValueError("Qwen API 模型名称无效")
        if not _ENV_NAME.fullmatch(result["api_key_env"]):
            raise ValueError("API Key 环境变量名称无效")
    if not 10 <= result["timeout_seconds"] <= 900:
        raise ValueError("Qwen API 超时必须在 10 至 900 秒")
    if not 1 <= result["max_attempts"] <= 5:
        raise ValueError("Qwen API 重试次数必须在 1 至 5")
    if not 2 <= result["max_frames"] <= 12:
        raise ValueError("Qwen API 骨架帧数必须在 2 至 12")
    return result


def load_qwen_api(path: Path) -> Dict[str, Any]:
    try:
        return normalize_qwen_api(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_QWEN_API)


def save_qwen_api(path: Path, value: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = normalize_qwen_api(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return cleaned


def load_qwen_api_secret(path: Path) -> str:
    """Read a server-local Web-entered key without ever serializing it."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_qwen_api_secret(path: Path, value: str) -> None:
    """Atomically store a Web-entered key with owner-only file permissions."""
    key = str(value or "").strip()
    if not key:
        raise ValueError("Qwen API Key 不能为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        # Windows does not provide POSIX mode bits.  The runtime directory's
        # ACL remains the deployment boundary in that case.
        pass
    temporary.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def clear_qwen_api_secret(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def resolve_api_key(config: Dict[str, Any], saved_key: str = "") -> Tuple[str, str]:
    """Environment wins; a protected Web-entered key is the fallback."""
    environment_key = os.environ.get(str(config.get("api_key_env") or ""), "").strip()
    if environment_key:
        return environment_key, "environment"
    saved_key = str(saved_key or "").strip()
    return (saved_key, "web_settings") if saved_key else ("", "")


def api_status(config: Dict[str, Any], saved_key: str = "") -> Dict[str, Any]:
    value = normalize_qwen_api(config)
    _key, credential_source = resolve_api_key(value, saved_key)
    key_present = bool(_key)
    return {
        **value,
        "key_present": key_present,
        "credential_source": credential_source or None,
        "available": bool(value["enabled"] and key_present),
        "reason": "" if value["enabled"] and key_present else (
            "未启用 Qwen API 备用通道" if not value["enabled"] else
            "未检测到环境变量 {}".format(value["api_key_env"])
        ),
    }


def _frame_data_urls(video: Path, max_frames: int) -> Tuple[List[str], Dict[str, Any]]:
    try:
        import cv2
        from dahua_cup.semantic_teacher.api.qwen3_backend import decode_video_frames
    except ImportError as exc:
        raise RuntimeError("Qwen API 骨架帧编码需要 OpenCV") from exc
    frames, provenance = decode_video_frames(video, max_frames)
    urls = []
    for frame in frames:
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not ok:
            raise RuntimeError("无法编码 Qwen API 骨架帧")
        urls.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"))
    return urls, provenance


def run_qwen_api(
    config: Dict[str, Any], *, sample_id: str, feature: Path,
    pose_video: Path, output: Path, prediction: Path | None = None,
    saved_key: str = "",
) -> Dict[str, Any]:
    """Call hosted Qwen with derived skeleton frames and validate teacher JSON."""
    status = api_status(config, saved_key)
    if not status["available"]:
        raise RuntimeError(status["reason"])
    key, _credential_source = resolve_api_key(config, saved_key)
    semantic_graph = summarize_pose_feature(sample_id, feature)
    student_distribution: dict[str, float] = {}
    if prediction is not None and Path(prediction).is_file():
        try:
            student = json.loads(Path(prediction).read_text(encoding="utf-8"))
            for item in student.get("topk", []):
                label = str(item.get("label") or "")
                if label in LABELS:
                    student_distribution[label] = float(item["score"])
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            # The teacher remains usable when a legacy or partial student
            # artifact cannot supply a Top-K hint.
            student_distribution = {}
    prompt = build_teacher_prompt(
        sample_id, semantic_graph, LABELS, "campus6", student_distribution
    )
    urls, frame_provenance = _frame_data_urls(Path(pose_video), status["max_frames"])
    content = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in urls
    ] + [{"type": "text", "text": prompt}]
    request_payload = {
        "model": status["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 900,
        "enable_thinking": False,
    }
    encoded = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    error: Optional[Exception] = None
    for attempt in range(1, status["max_attempts"] + 1):
        request = urllib.request.Request(
            status["endpoint"], data=encoded,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=status["timeout_seconds"]) as response:
                reply = json.loads(response.read().decode("utf-8"))
            message = reply["choices"][0]["message"]
            raw = str(message.get("content") or "").strip()
            if not raw:
                raise ValueError("Qwen API 返回了空内容")
            teacher, repairs = parse_teacher_output(extract_json_object(raw), LABELS)
            value = {
                "schema_version": "teacher_prediction.v1", "status": "completed",
                "task": "campus6", "label_space_size": len(LABELS),
                "model": status["model"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "result": teacher.to_dict(), "semantic_graph": semantic_graph,
                "provenance": {
                    "execution": "qwen_api_fallback", "prompt_version": PROMPT_VERSION,
                    "teacher_model_version": status["model"],
                    "endpoint": status["endpoint"], "request_id": reply.get("id") or reply.get("request_id"),
                    "usage": reply.get("usage", {}), "attempts": attempt,
                    "raw_response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "video_decode": frame_provenance, "api_key_env": status["api_key_env"],
                    "confidence_method": "teacher_self_report_uncalibrated",
                    "repairs": repairs or {},
                    "student_distribution": student_distribution,
                },
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
            return value
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < status["max_attempts"]:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError("Qwen API 调用或教师输出校验失败：{}".format(str(error)[:500]))
