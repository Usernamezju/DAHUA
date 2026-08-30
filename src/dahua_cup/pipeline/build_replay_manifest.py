"""Create a deterministic stratified replay JSONL for incremental Campus6 training."""

from __future__ import annotations

import argparse

from dahua_cup.pipeline.common import log_event, read_jsonl, write_jsonl
from dahua_cup.semantic_teacher.incremental.replay_buffer import (
    build_replay_buffer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument(
        "--strata",
        default="label,scene,camera,difficulty",
        help="Comma-separated deterministic replay strata",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    strata = tuple(value.strip() for value in args.strata.split(",") if value.strip())
    if not strata:
        raise ValueError("at least one replay stratum is required")
    selected = build_replay_buffer(
        read_jsonl(args.input_jsonl), args.capacity, strata=strata
    )
    for row in selected:
        row["label_source"] = "replay"
        row["split"] = "train"
    write_jsonl(args.output, selected)
    log_event(
        "replay_manifest_complete",
        selected=len(selected),
        capacity=args.capacity,
        strata=strata,
        output=args.output,
    )


if __name__ == "__main__":
    main()
