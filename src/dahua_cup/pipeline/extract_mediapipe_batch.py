"""Resumable MediaPipe-to-NTU25 extraction for a labeled video manifest.

Each process handles a deterministic shard, so several CPU workers can run on
the same manifest safely.  The actual pose conversion remains the established
``mediapipe_pose_worker.extract_video`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .mediapipe_pose_worker import extract_video


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--num-poses", type=int, choices=(1, 2), default=2)
    parser.add_argument("--delegate", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--joint-score-threshold", type=float, default=0.2)
    return parser.parse_args()


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSONL at %s:%d" % (path, line_number)) from exc
            if not row.get("video") or not row.get("label"):
                raise ValueError("manifest row %d needs video and label" % line_number)
            yield row


def _feature_path(output_dir, row):
    video = Path(row["video"])
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:12]
    return output_dir / str(row["label"]) / (video.stem + "__" + digest + ".npz")


def _completed_records(path):
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" or row.get("status") == "error":
            completed.add(str(row.get("video", "")))
    return completed


def main():
    args = parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index must be in [0, worker count)")
    if not args.input_manifest.is_file():
        raise FileNotFoundError(args.input_manifest)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    log_path = args.output_dir / "logs" / ("worker_%02d.jsonl" % args.worker_index)
    previously_completed = _completed_records(log_path) if args.resume else set()
    rows = [
        row for index, row in enumerate(_read_jsonl(args.input_manifest))
        if index % args.worker_count == args.worker_index
    ]
    print(
        "worker %d/%d: assigned=%d, prior_records=%d" % (
            args.worker_index, args.worker_count, len(rows), len(previously_completed)
        ),
        flush=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0, "skip": 0}
    with log_path.open("a", encoding="utf-8") as log:
        for position, row in enumerate(rows, 1):
            video = Path(row["video"])
            output = _feature_path(args.output_dir, row)
            if (
                args.resume
                and output.is_file()
                and (str(video) in previously_completed or not args.retry_errors)
            ):
                counts["skip"] += 1
                continue
            if args.resume and str(video) in previously_completed and not args.retry_errors:
                counts["skip"] += 1
                continue
            try:
                worker_args = argparse.Namespace(**vars(args))
                worker_args.video = str(video)
                worker_args.feature = str(output)
                # VIDEO mode owns a per-stream timestamp clock.  Creating a
                # fresh landmarker here deliberately resets that clock for the
                # next independent clip.
                arrays = extract_video(worker_args)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **arrays)
                event = {
                    "status": "ok", "video": str(video), "label": row["label"],
                    "feature": str(output), "frames": int(arrays["total_frames"]),
                    "valid_joints": int(np.count_nonzero(arrays["valid_mask"])),
                }
                counts["ok"] += 1
            except Exception as exc:
                event = {
                    "status": "error", "video": str(video), "label": row["label"],
                    "feature": str(output), "error": "%s: %s" % (type(exc).__name__, exc),
                }
                counts["error"] += 1
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            if position % 10 == 0 or position == len(rows):
                print(
                    "[%d/%d] ok=%d error=%d skip=%d" % (
                        position, len(rows), counts["ok"], counts["error"], counts["skip"]
                    ),
                    flush=True,
                )
    print("complete: %s" % counts, flush=True)


if __name__ == "__main__":
    main()
