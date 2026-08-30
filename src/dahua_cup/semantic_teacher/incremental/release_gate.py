"""Metric gates and recoverable production pointers for incremental releases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ReleaseGates:
    minimum_global_macro_f1_delta: float = 0.0
    minimum_new_macro_f1_delta: float = 0.0
    maximum_old_macro_f1_drop: float = 0.02
    minimum_dangerous_recall_delta: float = 0.0
    maximum_ece_increase: float = 0.02
    maximum_edge_size_bytes: int = 50 * 1024 * 1024


def evaluate_release(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    gates: ReleaseGates | None = None,
) -> dict:
    gates = gates or ReleaseGates()
    required = {
        "global_macro_f1",
        "new_macro_f1",
        "old_macro_f1",
        "dangerous_recall",
        "ece",
        "edge_size_bytes",
    }
    missing = sorted(required - candidate.keys())
    missing += sorted(
        f"baseline.{name}" for name in required - {"edge_size_bytes"} - baseline.keys()
    )
    if missing:
        raise ValueError("missing release metrics: " + ", ".join(missing))
    checks = {
        "global_macro_f1": (
            candidate["global_macro_f1"] - baseline["global_macro_f1"]
            >= gates.minimum_global_macro_f1_delta
        ),
        "new_macro_f1": (
            candidate["new_macro_f1"] - baseline["new_macro_f1"]
            >= gates.minimum_new_macro_f1_delta
        ),
        "old_forgetting": (
            baseline["old_macro_f1"] - candidate["old_macro_f1"]
            <= gates.maximum_old_macro_f1_drop
        ),
        "dangerous_recall": (
            candidate["dangerous_recall"] - baseline["dangerous_recall"]
            >= gates.minimum_dangerous_recall_delta
        ),
        "calibration": (
            candidate["ece"] - baseline["ece"] <= gates.maximum_ece_increase
        ),
        "edge_size": (
            int(candidate["edge_size_bytes"]) <= gates.maximum_edge_size_bytes
        ),
    }
    return {
        "schema_version": "release_gate.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "candidate": dict(candidate),
        "baseline": dict(baseline),
        "gates": asdict(gates),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


class ProductionPointer:
    """Atomic current/previous model pointer with explicit rollback history."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        if not self.path.is_file():
            return {
                "schema_version": "production_pointer.v1",
                "current_model_id": None,
                "previous_model_id": None,
                "history": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def promote(self, model_id: str, gate_report: Mapping) -> dict:
        if not model_id or not gate_report.get("passed"):
            raise ValueError("only a gate-passed model can be promoted")
        value = self.read()
        current = value.get("current_model_id")
        event = {
            "action": "promote",
            "from": current,
            "to": model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        value["previous_model_id"] = current
        value["current_model_id"] = model_id
        value.setdefault("history", []).append(event)
        self._write(value)
        return value

    def rollback(self, actor: str, reason: str) -> dict:
        value = self.read()
        previous = value.get("previous_model_id")
        current = value.get("current_model_id")
        if not previous:
            raise ValueError("there is no previous production model to restore")
        event = {
            "action": "rollback",
            "from": current,
            "to": previous,
            "actor": actor,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        value["current_model_id"] = previous
        value["previous_model_id"] = current
        value.setdefault("history", []).append(event)
        self._write(value)
        return value

    def _write(self, value: dict) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
