#!/usr/bin/env python3
"""
Campus6 video mining pipeline — full automated run.

Usage:
    python -m dahua_cup.dataset_construction.video_mining.run_pipeline            # full pipeline
    python -m dahua_cup.dataset_construction.video_mining.run_pipeline --download  # download only
    python -m dahua_cup.dataset_construction.video_mining.run_pipeline --slice     # slice only
    python -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen    # screen only
    python -m dahua_cup.dataset_construction.video_mining.continuous               # download + screen
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .download import run_download_phase
from .slice_clips import run_slice_phase
from .classify import run_screening_phase


def main():
    parser = argparse.ArgumentParser(description="Campus6 video mining pipeline")
    parser.add_argument("--download", action="store_true", help="Only run download phase")
    parser.add_argument("--slice", action="store_true", help="Only run slice phase")
    parser.add_argument("--screen", action="store_true", help="Only run screening phase")
    args = parser.parse_args()

    run_all = not (args.download or args.slice or args.screen)

    # ── Phase 1: Download ──────────────────────────────────────
    if run_all or args.download:
        print("=" * 60)
        print("PHASE 1: DOWNLOAD")
        print("=" * 60)
        downloaded = run_download_phase()
        total = sum(len(v) for v in downloaded.values())
        print(f"\nDownloaded {total} videos total.")
        for label, paths in downloaded.items():
            print(f"  {label}: {len(paths)}")
    else:
        # Load from disk
        downloaded = {}
        for label in config.KEYWORDS:
            label_dir = config.DOWNLOADS_DIR / label
            if label_dir.is_dir():
                downloaded[label] = sorted(label_dir.rglob("*.mp4"))
        print(f"Found {sum(len(v) for v in downloaded.values())} existing downloads")

    # ── Phase 2: Slice ─────────────────────────────────────────
    if run_all or args.slice:
        print("\n" + "=" * 60)
        print("PHASE 2: SLICE")
        print("=" * 60)
        all_clips = run_slice_phase(downloaded)
    else:
        all_clips = {}
        for label in config.KEYWORDS:
            label_dir = config.CLIPS_DIR / label
            if label_dir.is_dir():
                all_clips[label] = sorted(label_dir.rglob("*.mp4"))
        print(f"Found {sum(len(v) for v in all_clips.values())} existing clips")

    # ── Phase 3: Screen with the configured VLM ───────────────
    if run_all or args.screen:
        print("\n" + "=" * 60)
        print(f"PHASE 3: {config.CLASSIFIER_BACKEND.upper()} SCREENING")
        print("=" * 60)
        counts = run_screening_phase(all_clips)
        print("\nPipeline complete!")
        print(f"Results saved to: {config.SCREENED_DIR}")
    else:
        print("\nDone (screening skipped).")


if __name__ == "__main__":
    main()
