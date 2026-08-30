"""Run a configured batch closed loop without online weight updates."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--status-file", help="Web-readable incremental training heartbeat JSON")
    parser.add_argument("--estimated-total-seconds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_status(path, **values):
    if not path:
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "incremental_training_status.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


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
    configured_stages = [stage for stage in STAGES if stage in commands]
    started_at = datetime.now(timezone.utc).isoformat()
    _write_status(
        args.status_file,
        status="running",
        stage=configured_stages[0] if configured_stages else "completed",
        progress=0.0,
        started_at=started_at,
        pid=os.getpid(),
        estimated_total_seconds=args.estimated_total_seconds,
        sample_count=len(rows),
    )
    completed = 0
    try:
        for stage in STAGES:
            status = "planned" if stage in commands else "not_configured"
            if stage in commands:
                _write_status(
                    args.status_file,
                    status="running",
                    stage=stage,
                    progress=completed / max(1, len(configured_stages)),
                    started_at=started_at,
                    pid=os.getpid(),
                    estimated_total_seconds=args.estimated_total_seconds,
                    sample_count=len(rows),
                )
                command = render_command(
                    commands[stage],
                    base_model=args.base_model,
                    new_data=args.new_data,
                    output_manifest=args.output_manifest,
                )
                run_command(command, dry_run=args.dry_run)
                status = "dry_run" if args.dry_run else "completed"
                completed += 1
            run["stages"].append({"name": stage, "status": status})
    except Exception as exc:
        _write_status(
            args.status_file,
            status="failed",
            stage=stage,
            progress=completed / max(1, len(configured_stages)),
            started_at=started_at,
            pid=os.getpid(),
            message=f"{type(exc).__name__}: {exc}",
            sample_count=len(rows),
        )
        raise
    if not args.dry_run:
        destination = Path(args.output_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_status(
        args.status_file,
        status="completed",
        stage="completed",
        progress=1.0,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        estimated_remaining_seconds=0,
        sample_count=len(rows),
    )
    log_event("closed_loop_complete", stages=run["stages"], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
