"""Orchestrate video-to-feature extraction in the isolated pose environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import log_event, read_jsonl, render_command, require_file, run_command


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pose-command",
        help="Pose-env command template with {video}, {sample_id}, {output_dir}, {config}",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    require_file(args.config, "campus model config")
    rows = read_jsonl(args.input_manifest)
    if not args.pose_command and not args.dry_run:
        raise ValueError("--pose-command is required outside --dry-run (pose environment boundary)")
    output_dir = Path(args.output_dir)
    for index, row in enumerate(rows):
        video = require_file(row["video"], "input video")
        sample_id = str(row.get("sample_id") or video.stem)
        output_stem = output_dir / sample_id
        if args.resume and output_stem.with_suffix(".npz").exists() and output_stem.with_suffix(".json").exists():
            log_event("extract_skip", sample_id=sample_id, reason="artifact_exists")
            continue
        if args.pose_command:
            command = render_command(args.pose_command, video=video, sample_id=sample_id,
                                     output_dir=output_dir, config=args.config)
            run_command(command, dry_run=args.dry_run)
        else:
            log_event("extract_plan", index=index, sample_id=sample_id, video=str(video),
                      output_stem=str(output_stem), dry_run=True)
    log_event("extract_complete", samples=len(rows), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
