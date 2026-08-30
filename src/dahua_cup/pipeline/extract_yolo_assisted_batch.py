"""Resumable sharded batch runner for YOLO-assisted MediaPipe extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .mediapipe_yolo_assist import extract
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--yolo", type=Path, required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--worker-count", type=int, default=1)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = [r for i, r in enumerate(rows) if i % args.worker_count == args.worker_index]
    log_path = args.output_dir / "logs" / f"worker_{args.worker_index:02d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line).get("video") for line in log_path.read_text().splitlines()} if log_path.exists() else set()
    counts = {"ok": 0, "error": 0, "skip": 0}
    with log_path.open("a") as log:
        for position, row in enumerate(rows, 1):
            video = Path(row["video"])
            digest = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:12]
            feature = args.output_dir / row["label"] / f"{video.stem}__{digest}.npz"
            if str(video) in done and feature.is_file():
                counts["skip"] += 1
                continue
            try:
                arrays = extract(video, args.model, args.yolo)
                feature.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(feature, **arrays)
                event = {"status": "ok", "video": str(video), "label": row["label"], "feature": str(feature)}
                counts["ok"] += 1
            except Exception as error:
                event = {"status": "error", "video": str(video), "label": row["label"], "error": f"{type(error).__name__}: {error}"}
                counts["error"] += 1
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            if position % 5 == 0 or position == len(rows):
                print(f"[{position}/{len(rows)}] {counts}", flush=True)
    print(f"complete: {counts}", flush=True)


if __name__ == "__main__":
    main()
