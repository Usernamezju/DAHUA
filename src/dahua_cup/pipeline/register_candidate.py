"""Register a trained Campus6 checkpoint as an auditable release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dahua_cup.semantic_teacher.incremental.model_registry import ModelRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--latency-ms", required=True, type=float)
    parser.add_argument("--registry", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.latency_ms <= 0:
        raise ValueError("latency-ms must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    for path, description in (
        (checkpoint, "checkpoint"),
        (config, "config"),
        (metrics_path, "metrics"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["edge_size_bytes"] = checkpoint.stat().st_size
    record = ModelRegistry(args.registry).register(
        {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "checkpoint_path": str(checkpoint),
            "config_path": str(config),
            "config_hash": _sha256(config),
            "checkpoint_hash": _sha256(checkpoint),
            "metrics": metrics,
            "edge_size_bytes": metrics["edge_size_bytes"],
            "latency": {"device": "edge", "milliseconds": args.latency_ms},
        }
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
