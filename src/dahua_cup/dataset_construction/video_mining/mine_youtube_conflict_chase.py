#!/usr/bin/env python3
"""Build a local, auditable YouTube candidate pool for Campus6 conflict_chase.

The output consists of short *candidates* only.  Nothing is assumed to be a
correct label: ``manifest.csv`` retains its YouTube URL, query and title so it
can be reviewed before adding clips to the training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


QUERIES = [
    "CCTV two civilians chase after argument -police -car",
    "surveillance person runs away another person chases after fight -police",
    "home security camera neighbor argument chase two people",
    "store security camera two people altercation then chase",
    "CCTV footage two men chase after arguing -police",
    "CCTV footage two women chase after arguing -police",
    "security camera street argument one person chases another on foot",
    "surveillance footage civilian chase after physical altercation",
    "监控 两名男子 争执后追赶 无警察",
    "监控 两人 追打 跑动 全身",
    # Staged footage is deliberately included for a high-recall pose pool.
    # It remains an unreviewed candidate, never a ground-truth Campus6 label.
    "movie scene man chases another man after fight on foot",
    "movie scene woman chases another woman after argument",
    "action movie foot chase after confrontation two people",
    "film scene two people chase each other after fight",
    "TV drama scene person runs away another person chases",
    "short film argument then chase scene two people",
    "cinema scene fight then foot chase full scene",
    "电影片段 两个人 争执 追赶 跑步",
    "影视片段 追打 两人 跑动",
]

# Avoid clearly irrelevant or unsuitable (e.g. police/car/animal) results.
REJECT_TERMS = {
    "car chase", "vehicle", "truck", "motorcycle", "dog", "bear", "animal",
    "football", "soccer", "basketball", "baseball", "gameplay", "walkthrough",
    "trailer", "reaction", "compilation", "music video",
}


def run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with the local Windows proxy when configured."""
    env = os.environ.copy()
    proxy = env.get("CAMPUS6_YTDLP_PROXY", "http://172.21.224.1:7897")
    env["http_proxy"] = proxy
    env["https_proxy"] = proxy
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=env)


def ytdlp_prefix() -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--js-runtimes", "node:/usr/bin/node"]


def search(query: str, count: int) -> list[dict]:
    result = run(ytdlp_prefix() + [
        "--proxy", os.environ.get("CAMPUS6_YTDLP_PROXY", "http://172.21.224.1:7897"),
        f"ytsearch{count}:{query}", "--dump-single-json", "--flat-playlist", "--no-warnings",
    ], timeout=180)
    if result.returncode:
        print(f"[search failed] {query}: {result.stderr.splitlines()[-1:]}", file=sys.stderr)
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[search malformed] {query}", file=sys.stderr)
        return []
    return [entry for entry in payload.get("entries", []) if entry]


def acceptable(entry: dict) -> bool:
    duration = entry.get("duration")
    if not isinstance(duration, (int, float)) or duration < 8 or duration > 180:
        return False
    text = " ".join(str(entry.get(key, "")) for key in ("title", "description")).lower()
    return not any(term in text for term in REJECT_TERMS)


def discover(out_dir: Path, search_count: int, wanted_raw: int) -> list[dict]:
    selected: dict[str, dict] = {}
    # Spread selection across queries.  A plain global limit otherwise fills
    # almost entirely from the first search result page and yields a brittle,
    # narrow source distribution.
    per_query_limit = math.ceil(wanted_raw / len(QUERIES))
    for query in QUERIES:
        print(f"searching: {query}", flush=True)
        added = 0
        for entry in search(query, search_count):
            identifier = str(entry.get("id", ""))
            if identifier and identifier not in selected and acceptable(entry):
                selected[identifier] = {
                    "youtube_id": identifier,
                    "url": entry.get("url") or f"https://www.youtube.com/watch?v={identifier}",
                    "title": str(entry.get("title", "")),
                    "duration": float(entry["duration"]),
                    "query": query,
                }
                added += 1
                if added >= per_query_limit:
                    break
    rows = list(selected.values())
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifests" / "discovered_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return rows


def download(item: dict, raw_dir: Path, archive: Path) -> Path | None:
    identifier = item["youtube_id"]
    existing = sorted(raw_dir.glob(f"{identifier}.*"))
    if existing and existing[0].stat().st_size > 10_000:
        return existing[0]
    result = run(ytdlp_prefix() + [
        "--proxy", os.environ.get("CAMPUS6_YTDLP_PROXY", "http://172.21.224.1:7897"),
        item["url"], "--no-playlist", "--no-warnings", "--restrict-filenames",
        "--match-filter", "duration >= 8 & duration <= 180",
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4", "--download-archive", str(archive),
        "-o", str(raw_dir / "%(id)s.%(ext)s"), "--print", "after_move:filepath",
    ], timeout=420)
    if result.returncode:
        tail = "; ".join(result.stderr.strip().splitlines()[-2:])
        print(f"[download failed] {identifier}: {tail}", file=sys.stderr)
        return None
    for line in reversed(result.stdout.splitlines()):
        path = Path(line.strip())
        if path.is_file() and path.stat().st_size > 10_000:
            return path
    existing = sorted(raw_dir.glob(f"{identifier}.*"))
    return existing[0] if existing and existing[0].stat().st_size > 10_000 else None


def duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True, capture_output=True, timeout=30, check=True)
    return float(result.stdout.strip())


def starts(total: float, per_source: int) -> list[float]:
    usable = total - 8.0
    if usable <= 0.25:
        return [0.0]
    count = min(per_source, max(1, math.floor(total / 8)))
    return [round(usable * index / (count - 1), 2) if count > 1 else round(usable / 2, 2)
            for index in range(count)]


def cut(source: Path, clip: Path, start: float) -> bool:
    if clip.is_file() and clip.stat().st_size > 10_000:
        return True
    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(source),
        "-t", "8", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", str(clip),
    ], text=True, capture_output=True, timeout=180)
    if result.returncode or not clip.is_file() or clip.stat().st_size <= 10_000:
        clip.unlink(missing_ok=True)
        print(f"[cut failed] {source.name}@{start:.2f}: {result.stderr[-300:]}", file=sys.stderr)
        return False
    return True


def read_clip_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def dense_starts(total: float, step: float) -> list[float]:
    """Cover a source with overlapping 8s candidates, without any >8s clip."""
    final_start = total - 8.0
    if final_start <= 0.25:
        return [0.0]
    values, start = [], 0.0
    while start < final_start - 0.01:
        values.append(round(start, 2))
        start += step
    values.append(round(final_start, 2))
    return values


def slice_existing(args: argparse.Namespace, out_dir: Path) -> None:
    """Top up an interrupted source set without pretending failed downloads worked."""
    manifest_path = out_dir / "manifests" / "clip_manifest.csv"
    rows = read_clip_rows(manifest_path)
    known = {row["clip_file"] for row in rows}
    seen_segments = {
        (row.get("youtube_id", ""), round(float(row.get("start_seconds", "nan")), 2))
        for row in rows
        if row.get("youtube_id") and row.get("start_seconds", "")
    }
    fieldnames = ["clip_file", "youtube_id", "source_url", "source_title", "query", "start_seconds", "duration_seconds", "candidate_label", "review_status"]
    source_meta = {
        str(item.get("youtube_id", "")): item
        for item in json.loads((out_dir / "manifests" / "discovered_sources.json").read_text(encoding="utf-8"))
    }
    raw_dir, clips_dir = out_dir / "raw_videos", out_dir / "clips_8s"
    for source in sorted(raw_dir.glob("*.mp4")):
        if len(rows) >= args.max_clips:
            break
        youtube_id = source.name.split(".", 1)[0]
        try:
            source_duration = duration(source)
        except Exception as exc:
            print(f"[probe failed] {source.name}: {exc}", file=sys.stderr)
            continue
        item = source_meta.get(youtube_id, {
            "youtube_id": youtube_id, "url": f"https://www.youtube.com/watch?v={youtube_id}",
            "title": "", "query": "previous successful download",
        })
        for index, start in enumerate(dense_starts(source_duration, args.dense_step)):
            if len(rows) >= args.max_clips:
                break
            clip = clips_dir / f"{youtube_id}_s{start:06.2f}_d{index:03d}.mp4"
            relative = str(clip.relative_to(out_dir))
            if relative in known or (youtube_id, start) in seen_segments:
                continue
            if cut(source, clip, start):
                rows.append({
                    "clip_file": relative, "youtube_id": youtube_id, "source_url": item["url"],
                    "source_title": item["title"], "query": item["query"],
                    "start_seconds": f"{start:.2f}", "duration_seconds": "8.00",
                    "candidate_label": "conflict_chase", "review_status": "unreviewed",
                })
                known.add(relative)
                seen_segments.add((youtube_id, start))
        print(f"dense slicing: {source.name}; clips={len(rows)}", flush=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out_dir), "clips": len(rows), "mode": "slice_existing"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wanted-raw", type=int, default=55)
    parser.add_argument("--search-count", type=int, default=40)
    parser.add_argument("--per-source", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=150)
    parser.add_argument("--slice-existing", action="store_true", help="only densely re-slice already downloaded MP4s")
    parser.add_argument("--dense-step", type=float, default=3.5, help="seconds between starts for --slice-existing")
    args = parser.parse_args()
    out_dir = args.output.resolve()
    raw_dir, clips_dir = out_dir / "raw_videos", out_dir / "clips_8s"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    if args.slice_existing:
        slice_existing(args, out_dir)
        return
    items = discover(out_dir, args.search_count, args.wanted_raw)
    print(f"discovered {len(items)} candidate sources", flush=True)
    downloaded, clip_rows = 0, []
    for item in items:
        source = download(item, raw_dir, out_dir / "manifests" / "yt_dlp_archive.txt")
        if not source:
            continue
        downloaded += 1
        try:
            source_duration = duration(source)
        except Exception as exc:
            print(f"[probe failed] {source.name}: {exc}", file=sys.stderr)
            continue
        for index, start in enumerate(starts(source_duration, args.per_source)):
            if len(clip_rows) >= args.max_clips:
                break
            clip = clips_dir / f"{item['youtube_id']}_s{start:06.2f}_c{index:02d}.mp4"
            if cut(source, clip, start):
                clip_rows.append({
                    "clip_file": str(clip.relative_to(out_dir)), "youtube_id": item["youtube_id"],
                    "source_url": item["url"], "source_title": item["title"], "query": item["query"],
                    "start_seconds": f"{start:.2f}", "duration_seconds": "8.00",
                    "candidate_label": "conflict_chase", "review_status": "unreviewed",
                })
        print(f"downloaded={downloaded}; clips={len(clip_rows)}; latest={source.name}", flush=True)
        if len(clip_rows) >= args.max_clips:
            break
        time.sleep(0.2)
    fieldnames = ["clip_file", "youtube_id", "source_url", "source_title", "query", "start_seconds", "duration_seconds", "candidate_label", "review_status"]
    with (out_dir / "manifests" / "clip_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clip_rows)
    print(json.dumps({"output": str(out_dir), "downloaded_sources": downloaded, "clips": len(clip_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
