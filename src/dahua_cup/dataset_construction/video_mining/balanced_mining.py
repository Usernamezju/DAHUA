"""Download a resumable, balanced batch of Campus6 raw candidate videos.

The batch target is per *class*, never per keyword.  Successful downloads are
recorded immediately so a reboot resumes toward the same target without
redownloading earlier videos.  This module does not call any VLM API.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import config
from .download import download_keyword
from .slice_clips import slice_video


TARGET_LABELS = (
    "normal_walk",
    "normal_run",
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
)
BATCH_MANIFEST = config.DATA_ROOT / "balanced_mining_50_manifest.jsonl"


def _load_completed() -> tuple[Counter, set[str]]:
    counts: Counter = Counter()
    paths: set[str] = set()
    if not BATCH_MANIFEST.is_file():
        return counts, paths
    for line in BATCH_MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            label = str(row["label"])
            path = str(row["path"])
        except (TypeError, ValueError, KeyError):
            continue
        if label in TARGET_LABELS and path not in paths:
            counts[label] += 1
            paths.add(path)
    return counts, paths


def _append_download(label: str, keyword: str, path: Path) -> None:
    BATCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_MANIFEST.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "label": label,
            "keyword": keyword,
            "path": str(path.resolve()),
        }, ensure_ascii=False) + "\n")


def _pending_unsliced() -> list[tuple[str, Path]]:
    pending: list[tuple[str, Path]] = []
    for label in TARGET_LABELS:
        raw_dir = config.DOWNLOADS_DIR / label
        if not raw_dir.is_dir():
            continue
        for video in sorted(raw_dir.rglob("*.mp4")):
            first_clip = config.CLIPS_DIR / label / f"{video.stem}_clip0000.mp4"
            if not first_clip.is_file() or first_clip.stat().st_size <= 500:
                pending.append((label, video))
    return pending


def run(target_per_class: int) -> int:
    completed, known_paths = _load_completed()
    new_paths: list[tuple[str, Path]] = []

    for label in TARGET_LABELS:
        keywords = config.KEYWORDS[label]
        attempts = 0
        max_attempts = max(target_per_class * 8, len(keywords) * 12)
        cursor = 0
        print(f"\n[download] {label}: {completed[label]}/{target_per_class}", flush=True)
        while completed[label] < target_per_class and attempts < max_attempts:
            keyword = keywords[cursor % len(keywords)]
            cursor += 1
            attempts += 1
            paths = download_keyword(keyword, label, max_downloads=1)
            for path in paths:
                resolved = str(path.resolve())
                if resolved in known_paths:
                    continue
                known_paths.add(resolved)
                completed[label] += 1
                new_paths.append((label, path))
                _append_download(label, keyword, path)
                # Make each candidate available to the independent screening
                # worker immediately instead of waiting for the full batch.
                clips = slice_video(path, config.CLIPS_DIR / label)
                print(
                    f"[download] {label}: {completed[label]}/{target_per_class}; "
                    f"sliced {len(clips)} clips",
                    flush=True,
                )
                if completed[label] >= target_per_class:
                    break
        if completed[label] < target_per_class:
            print(
                f"[download] {label}: stopped at {completed[label]}/{target_per_class} "
                f"after {attempts} keyword attempts; rerun to continue.",
                flush=True,
            )

    # Recover interrupted earlier downloads that have no clip series.  New
    # downloads above are sliced immediately, so this normally has no work.
    to_slice = _pending_unsliced()
    print(f"\n[slice] {len(to_slice)} newly available raw videos", flush=True)
    for number, (label, path) in enumerate(to_slice, 1):
        clips = slice_video(path, config.CLIPS_DIR / label)
        print(f"[slice] {number}/{len(to_slice)} {path.name}: {len(clips)} clips", flush=True)

    print("\n[summary] " + ", ".join(
        f"{label}={completed[label]}/{target_per_class}" for label in TARGET_LABELS
    ), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Balanced Campus6 candidate mining")
    parser.add_argument("--target-per-class", type=int, default=50)
    args = parser.parse_args(argv)
    if args.target_per_class < 1:
        parser.error("target-per-class must be positive")
    return run(args.target_per_class)


if __name__ == "__main__":
    raise SystemExit(main())
