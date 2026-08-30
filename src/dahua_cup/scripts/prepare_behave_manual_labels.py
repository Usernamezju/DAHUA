#!/usr/bin/env python3
"""Cut BEHAVE interaction intervals into short manual-label candidates."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


FPS = 25.0
SOURCE_LABELS = {"WalkTogether", "RunTogether", "Chase", "Fight"}
COARSE_SUGGESTIONS = {
    "WalkTogether": "normal_walk",
    "RunTogether": "normal_run",
    "Chase": "chase_unspecified",
    "Fight": "conflict_unspecified",
}
ANNOTATION_PATTERN = re.compile(
    r".*?;\s*(?P<start>\d+)\s*;\s*(?P<end>\d+)\s*;\s*(?P<label>[A-Za-z]+)"
)
VIDEO_PATTERN = re.compile(r"(?P<start>\d+)-(?P<end>\d+)\.avi$")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--padding-seconds", type=float, default=0.25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--codec",
        default="libx264",
        choices=("libx264", "libopenh264", "mpeg4"),
    )
    return parser


def discover_videos(source_dir):
    videos = []
    for path in sorted(Path(source_dir).glob("*.avi")):
        match = VIDEO_PATTERN.fullmatch(path.name)
        if match:
            videos.append((int(match["start"]), int(match["end"]), path))
    if not videos:
        raise FileNotFoundError(f"no frame-range AVI files found under {source_dir}")
    return videos


def parse_intervals(markup_path, videos, min_seconds):
    intervals = set()
    for line in Path(markup_path).read_text(encoding="utf-8").splitlines():
        match = ANNOTATION_PATTERN.match(line)
        if not match:
            continue
        start, end = int(match["start"]), int(match["end"])
        label = match["label"]
        if label not in SOURCE_LABELS or end <= start:
            continue
        if (end - start) / FPS < min_seconds:
            continue
        for video_start, video_end, video_path in videos:
            if video_start <= start and end <= video_end:
                intervals.add((label, start, end, video_start, video_path))
                break
    return sorted(intervals, key=lambda item: (item[1], item[2], item[0]))


def cut_clip(
    source,
    destination,
    start_seconds,
    duration_seconds,
    *,
    ffmpeg="ffmpeg",
    codec="libx264",
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.3f}",
        "-an",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        command.extend(("-preset", "veryfast", "-crf", "20"))
    elif codec == "libopenh264":
        command.extend(("-b:v", "2M", "-maxrate", "2M", "-bufsize", "4M"))
    else:
        command.extend(("-q:v", "3"))
    command.append(str(destination))
    subprocess.run(command, check=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    videos = discover_videos(source_dir)
    intervals = parse_intervals(source_dir / "markup.txt", videos, args.min_seconds)

    rows = []
    for label, start, end, video_start, video_path in intervals:
        local_start = max(0.0, (start - video_start) / FPS - args.padding_seconds)
        duration = (end - start) / FPS + 2 * args.padding_seconds
        clip_id = f"behave_{label.lower()}_{start}_{end}"
        destination = output_dir / "clips" / label / f"{clip_id}.mp4"
        cut_clip(
            video_path,
            destination,
            local_start,
            duration,
            ffmpeg=args.ffmpeg,
            codec=args.codec,
        )
        rows.append(
            {
                "clip_id": clip_id,
                "path": str(destination),
                "source_dataset": "BEHAVE",
                "source_label": label,
                "source_start_frame": start,
                "source_end_frame": end,
                "suggested_coarse_label": COARSE_SUGGESTIONS[label],
                "manual_label": "",
                "notes": "",
            }
        )

    if not rows:
        raise RuntimeError(
            "no WalkTogether/RunTogether/Chase/Fight intervals were prepared"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "candidate_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} clips and {manifest}")


if __name__ == "__main__":
    main()
