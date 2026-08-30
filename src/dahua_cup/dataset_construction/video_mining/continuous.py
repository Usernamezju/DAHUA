"""Concurrent local producer/consumer runner for Campus6 video mining.

The producer downloads a bounded number of videos for each keyword and slices
them into ``clips/<retrieval_hint>/``.  The consumer independently scans every
clip directory and submits only clips absent from ``screening_manifest.jsonl``.
Both sides are restart-safe.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from . import config
from .classify import RateLimitExceeded, run_screening_phase
from .download import run_download_phase
from .slice_clips import slice_video


def _discover_downloads(root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}
    for label in config.KEYWORDS:
        directory = root / label
        discovered[label] = sorted(directory.rglob("*.mp4")) if directory.is_dir() else []
    return discovered


def _discover_clips() -> dict[str, list[Path]]:
    """Consume every clip below clips/, including manually added test folders."""

    if not config.CLIPS_DIR.is_dir():
        return {"available": []}
    return {"available": sorted(config.CLIPS_DIR.rglob("*.mp4"))}


def _slice_all_downloads() -> int:
    raw_videos = _discover_downloads(config.DOWNLOADS_DIR)
    total = 0
    for label, videos in raw_videos.items():
        output_dir = config.CLIPS_DIR / label
        for video in videos:
            total += len(slice_video(video, output_dir))
    return total


def _producer(rounds: int, max_per_keyword: int, stop: threading.Event) -> None:
    round_number = 0
    while not stop.is_set() and (rounds == 0 or round_number < rounds):
        round_number += 1
        print(f"\n[producer] download round {round_number}", flush=True)
        downloaded = run_download_phase(max_per_keyword=max_per_keyword)
        downloaded_count = sum(len(paths) for paths in downloaded.values())
        clip_count = _slice_all_downloads()
        print(
            f"[producer] round {round_number}: {downloaded_count} new videos; "
            f"{clip_count} total verified clips available",
            flush=True,
        )
    print("[producer] finished", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Continuously mine and screen Campus6 clips")
    parser.add_argument(
        "--producer-rounds",
        type=int,
        default=1,
        help="Download rounds; 0 means continue indefinitely (default: 1)",
    )
    parser.add_argument(
        "--max-downloads-per-keyword",
        type=int,
        default=1,
        help="New videos attempted for each keyword per producer round (default: 1)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Delay before rescanning clips after the current backlog is drained",
    )
    args = parser.parse_args(argv)
    if args.producer_rounds < 0 or args.max_downloads_per_keyword < 1 or args.poll_seconds < 1:
        parser.error("rounds must be >= 0; downloads and poll seconds must be positive")

    stop = threading.Event()
    producer = threading.Thread(
        target=_producer,
        args=(args.producer_rounds, args.max_downloads_per_keyword, stop),
        name="campus6-download-producer",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            clips = _discover_clips()
            total = sum(len(paths) for paths in clips.values())
            print(f"\n[consumer] scanned {total} clips", flush=True)
            try:
                run_screening_phase(clips)
            except RateLimitExceeded:
                backoff = config.MODELSCOPE_QUOTA_BACKOFF_SECONDS
                print(f"[consumer] provider quota reached; backing off for {backoff}s", flush=True)
                time.sleep(backoff)
                continue
            print(f"[consumer] backlog drained; waiting {args.poll_seconds}s", flush=True)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[continuous] stopping", flush=True)
        return 130
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
