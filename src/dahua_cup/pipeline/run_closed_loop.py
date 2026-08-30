"""Run a configured batch closed loop without online weight updates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import file_hash, log_event, read_jsonl, render_command, run_command


STAGES = ("extract", "mine", "teacher", "filter", "train", "validate")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--new-data", required=True, help="JSONL batch manifest")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--stage-command", action="append", default=[], help="stage=command template")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = read_jsonl(args.new_data)
    commands = {}
    for item in args.stage_command:
        stage, separator, command = item.partition("=")
        if not separator or stage not in STAGES:
            raise ValueError(f"--stage-command must be one of {STAGES}: {item}")
        commands[stage] = command
    run = {
        "schema_version": "closed_loop_run.v1",
        "base_model": args.base_model,
        "new_data": str(Path(args.new_data).resolve()),
        "new_data_hash": file_hash(args.new_data),
        "sample_count": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }
    for stage in STAGES:
        status = "planned" if stage in commands else "not_configured"
        if stage in commands:
            command = render_command(commands[stage], base_model=args.base_model,
                                     new_data=args.new_data, output_manifest=args.output_manifest)
            run_command(command, dry_run=args.dry_run)
            status = "dry_run" if args.dry_run else "completed"
        run["stages"].append({"name": stage, "status": status})
    if not args.dry_run:
        destination = Path(args.output_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_event("closed_loop_complete", stages=run["stages"], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
