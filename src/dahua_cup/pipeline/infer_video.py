"""Orchestrate pose extraction and student inference for one video."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import log_event, render_command, require_file, run_command


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--pose-command", required=True)
    parser.add_argument("--student-command", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    video = require_file(args.video, "input video")
    work_dir = Path(args.work_dir)
    feature = work_dir / f"{video.stem}.npz"
    common = {"video": video, "feature": feature, "output": Path(args.output), "work_dir": work_dir}
    run_command(render_command(args.pose_command, **common), dry_run=args.dry_run)
    run_command(render_command(args.student_command, **common), dry_run=args.dry_run)
    log_event("infer_complete", video=str(video), output=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
