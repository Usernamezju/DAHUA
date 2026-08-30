"""Append-only JSONL model registry with guarded release transitions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


TRANSITIONS = {
    "candidate": {"validated", "archived"},
    "validated": {"canary", "archived"},
    "canary": {"production", "archived"},
    "production": {"archived"},
    "archived": set(),
}


class ModelRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def register(self, record):
        value = dict(record)
        required = {"model_id", "dataset_id", "config_hash", "checkpoint_hash", "metrics", "edge_size_bytes", "latency"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"missing registry fields: {', '.join(missing)}")
        if any(row["model_id"] == value["model_id"] for row in self.records()):
            raise ValueError("model_id already exists")
        value["status"] = "candidate"
        value["created_at"] = datetime.now(timezone.utc).isoformat()
        self._append(value)
        return value

    def transition(self, model_id, new_status):
        history = [row for row in self.records() if row["model_id"] == model_id]
        if not history:
            raise KeyError(model_id)
        current = history[-1]["status"]
        if new_status not in TRANSITIONS[current]:
            raise ValueError(f"invalid model transition: {current} -> {new_status}")
        event = dict(history[-1], status=new_status, previous_status=current,
                     updated_at=datetime.now(timezone.utc).isoformat())
        self._append(event)
        return event

    def _append(self, value):
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
