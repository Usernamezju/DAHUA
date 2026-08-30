"""
FFmpeg video slicer — cut long videos into short overlapping clips.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

from . import config


def _ffprobe_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    return float(proc.stdout.strip())


def slice_video(video_path: Path, output_dir: Path, clip_duration: float = None,
                overlap: float = None) -> List[Path]:
    """Slice a video into clips, return list of clip paths.

    Uses ffmpeg -c copy (stream copy) for fast lossless slicing.
    Falls back to re-encode if stream copy produces broken files.
    """
    if clip_duration is None:
        clip_duration = config.CLIP_DURATION
    if overlap is None:
        overlap = config.CLIP_OVERLAP

    try:
        total_duration = _ffprobe_duration(video_path)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        print(f"  [skip] cannot probe duration: {video_path.name} ({exc})", file=sys.stderr)
        return []

    if total_duration < config.MIN_CLIP_DURATION:
        print(f"  [skip] too short ({total_duration:.1f}s): {video_path.name}", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    clips: List[Path] = []
    start = 0.0
    idx = 0
    step = clip_duration - overlap

    while start + config.MIN_CLIP_DURATION <= total_duration:
        out_path = output_dir / f"{stem}_clip{idx:04d}.mp4"
        # Continuous mining repeatedly scans the raw directory.  A verified
        # existing segment is already a completed unit of work, so never
        # re-encode it just because a later producer pass sees its source.
        if out_path.is_file() and out_path.stat().st_size > 500:
            try:
                if _ffprobe_duration(out_path) >= config.MIN_CLIP_DURATION:
                    clips.append(out_path)
                    start += step
                    idx += 1
                    continue
            except (ValueError, subprocess.TimeoutExpired):
                pass
        # Try stream copy first (fast), then re-encode if needed
        for attempt in range(2):
            if attempt == 0:
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{start:.2f}",
                    "-i", str(video_path),
                    "-t", f"{clip_duration:.2f}",
                    "-c", "copy",
                    str(out_path),
                ]
            else:
                # Re-encode with fast preset if stream copy fails
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{start:.2f}",
                    "-i", str(video_path),
                    "-t", f"{clip_duration:.2f}",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "28",
                    str(out_path),
                ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 500:
                # Verify duration
                try:
                    dur = _ffprobe_duration(out_path)
                    if dur >= config.MIN_CLIP_DURATION:
                        clips.append(out_path)
                        break
                except (ValueError, subprocess.TimeoutExpired):
                    pass
            # Remove failed file, try next approach
            if out_path.exists():
                out_path.unlink()

        start += step
        idx += 1

    return clips


def run_slice_phase(downloaded: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Slice all downloaded videos into clips.

    Returns dict mapping label → list of clip Paths.
    """
    all_clips: dict[str, list[Path]] = {}
    total_videos = sum(len(paths) for paths in downloaded.values())

    for label, paths in downloaded.items():
        out_dir = config.CLIPS_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        all_clips[label] = []
        for path in paths:
            clips = slice_video(path, out_dir)
            all_clips[label].extend(clips)
            if clips:
                print(f"  {path.stem}: {len(clips)} clips")

    total_clips = sum(len(v) for v in all_clips.values())
    print(f"\nTotal: {total_videos} videos → {total_clips} clips")
    return all_clips


def run_slice_phase_single(video_path: Path, output_dir: Path) -> List[Path]:
    """Slice a single video file (for manual use)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return slice_video(video_path, output_dir)
