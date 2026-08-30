"""Minimal hosted multimodal API clients with retries and no vendor SDK."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ENDPOINTS = {
    "zhipu": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


class MissingAPIKey(RuntimeError):
    pass


class HostedModelError(RuntimeError):
    pass


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class HostedVisionClient:
    def __init__(self, provider_config: Dict[str, Any], request_config: Dict[str, Any]):
        self.config = provider_config
        self.request_config = request_config
        self.provider = str(provider_config["provider"])
        self.model = str(provider_config["model"])
        key_name = str(provider_config["api_key_env"])
        self.api_key = os.environ.get(key_name, "").strip()
        self.api_key_env = key_name
        if self.provider not in ENDPOINTS:
            raise ValueError(f"unsupported provider: {self.provider}")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def require_key(self):
        if not self.available:
            raise MissingAPIKey(
                f"{self.model} requires environment variable {self.api_key_env}"
            )

    def _payload(self, sheet_paths: Sequence[Path], prompt: str) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for path in sheet_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(path)},
                }
            )
        content.append({"type": "text", "text": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(self.request_config.get("temperature", 0.0)),
            "max_tokens": int(self.config.get("max_tokens", 1800)),
        }
        if self.provider == "zhipu":
            payload["thinking"] = {
                "type": "enabled" if self.config.get("thinking") else "disabled"
            }
        elif self.provider == "dashscope" and "thinking" in self.config:
            payload["enable_thinking"] = bool(self.config["thinking"])
        # Gemini: no thinking/reasoning config — pass through as-is
        return payload

    def invoke(self, sheet_paths: Sequence[Path], prompt: str) -> Dict[str, Any]:
        self.require_key()
        payload = self._payload(sheet_paths, prompt)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempts = int(self.request_config.get("max_attempts", 5))
        backoff = float(self.request_config.get("initial_backoff_seconds", 3))
        timeout = float(self.request_config.get("timeout_seconds", 240))
        last_error = None

        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                ENDPOINTS[self.provider],
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8")
                result = json.loads(body)
                message = result["choices"][0]["message"]
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise HostedModelError(f"{self.model} returned empty content")
                return {
                    "provider": self.provider,
                    "model": self.model,
                    "content": content,
                    "reasoning_content": message.get("reasoning_content", ""),
                    "usage": result.get("usage", {}),
                    "request_id": result.get("id") or result.get("request_id"),
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = HostedModelError(
                    f"{self.model} HTTP {exc.code}: {detail[:600]}"
                )
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, HostedModelError) as exc:
                last_error = exc
            if attempt < attempts:
                time.sleep(backoff * (2 ** (attempt - 1)))

        raise HostedModelError(str(last_error or f"{self.model} request failed"))
