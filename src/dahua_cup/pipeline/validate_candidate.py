"""Evaluate metric gates and optionally promote an incremental Campus6 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dahua_cup.semantic_teacher.incremental.model_registry import ModelRegistry
from dahua_cup.semantic_teacher.incremental.release_gate import (
    ProductionPointer,
    ReleaseGates,
    evaluate_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-registry")
    parser.add_argument("--production-pointer")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--max-old-drop", type=float, default=0.02)
    parser.add_argument("--max-ece-increase", type=float, default=0.02)
    return parser


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    candidate_metrics = _json(args.candidate_metrics)
    registry = None
    current = None
    if args.model_id and args.model_registry:
        registry = ModelRegistry(args.model_registry)
        history = [
            row for row in registry.records()
            if row["model_id"] == args.model_id
        ]
        if not history:
            raise KeyError(args.model_id)
        current = history[-1]
        # Model size is derived from the registered checkpoint, never trusted
        # from a hand-written metrics JSON.
        candidate_metrics["edge_size_bytes"] = current["edge_size_bytes"]
    report = evaluate_release(
        candidate_metrics,
        _json(args.baseline_metrics),
        ReleaseGates(
            maximum_old_macro_f1_drop=args.max_old_drop,
            maximum_ece_increase=args.max_ece_increase,
        ),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.promote:
        if not all(
            (args.model_id, args.model_registry, args.production_pointer)
        ):
            raise ValueError(
                "promotion requires model-id, model-registry and production-pointer"
            )
        if not report["passed"]:
            raise ValueError("candidate failed release gates")
        if registry is None or current is None:
            raise ValueError("promotion model was not resolved from registry")
        if current["status"] == "candidate":
            current = registry.transition(args.model_id, "validated")
        if current["status"] == "validated":
            current = registry.transition(args.model_id, "canary")
        if current["status"] != "canary":
            raise ValueError("model must be candidate, validated or canary")
        registry.transition(args.model_id, "production")
        ProductionPointer(args.production_pointer).promote(
            args.model_id, report
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
