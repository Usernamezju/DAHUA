"""
yt-dlp video downloader — search keywords and download matching videos.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from . import config

YTDLP_BIN = "yt-dlp"

# yt-dlp needs a JS runtime for YouTube extraction
YTDLP_BASE_ARGS = ["--js-runtimes", "node:/usr/bin/node"]


def _run_ytdlp(args: List[str], timeout: int = 300) -> int:
    """Run yt-dlp with proxy env, return exit code."""
    env = os.environ.copy()
    env["https_proxy"] = config.PROXY_URL
    env["http_proxy"] = config.PROXY_URL
    proc = subprocess.run(
        [YTDLP_BIN] + args,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().splitlines()[-3:]
        print(f"  [yt-dlp] exit {proc.returncode}: {'; '.join(stderr_tail)}",
              file=sys.stderr)
    return proc.returncode


def search_videos(keyword: str, count: int = None) -> List[str]:
    """Search YouTube/Bilibili and return video URLs.

    Uses yt-dlp's built-in search with `ytsearchN:keyword` syntax.
    """
    if count is None:
        count = config.YTDLP_SEARCH_COUNT
    query = f"ytsearch{count}:{keyword}"

    print(f"  Searching: {query}")
    # --dump-json for reliable structured output
    args = YTDLP_BASE_ARGS + [
        query,
        "--dump-json",
        "--no-playlist",
        "--ignore-errors",
        "--match-filter", f"duration < {config.YTDLP_MAX_DURATION}",
    ]
    env = os.environ.copy()
    env["https_proxy"] = config.PROXY_URL
    env["http_proxy"] = config.PROXY_URL
    try:
        proc = subprocess.run(
            [YTDLP_BIN] + args,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"    [yt-dlp] search timed out; skipping keyword: {keyword}", file=sys.stderr)
        return []
    urls: List[str] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            url = info.get("webpage_url") or info.get("url") or ""
            if url and url.startswith("http"):
                urls.append(url)
        except json.JSONDecodeError:
            pass
    for line in proc.stderr.strip().splitlines():
        if "ERROR" in line:
            print(f"    {line}", file=sys.stderr)
    return urls


def download_video(url: str, output_dir: Path, prefix: str) -> Path | None:
    """Download a single video, return output path or None on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / f"{prefix}_%(id)s.%(ext)s")
    args = YTDLP_BASE_ARGS + [
        url,
        "-o", template,
        "-f", f"best[ext={config.YTDLP_FORMAT}]/best",
        "--no-playlist",
        "--match-filter", f"duration < {config.YTDLP_MAX_DURATION}",
        "--ignore-errors",
        "--no-warnings",
        "--no-overwrites",
        "--download-archive", str(config.YTDLP_ARCHIVE_FILE),
    ]
    env = os.environ.copy()
    env["https_proxy"] = config.PROXY_URL
    env["http_proxy"] = config.PROXY_URL
    try:
        proc = subprocess.run(
            [YTDLP_BIN] + args,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.YTDLP_MAX_DURATION + 120,
        )
    except subprocess.TimeoutExpired:
        print(f"    [yt-dlp] download timed out; skipping URL", file=sys.stderr)
        return None

    # Find the downloaded file by scanning output for "Destination: ..."
    for line in proc.stdout.strip().splitlines():
        if "Destination:" in line:
            dest = line.split("Destination:", 1)[1].strip()
            p = Path(dest)
            if p.exists() and p.stat().st_size > 1000:
                return p
    for line in proc.stderr.strip().splitlines():
        if "Destination:" in line:
            dest = line.split("Destination:", 1)[1].strip()
            p = Path(dest)
            if p.exists() and p.stat().st_size > 1000:
                return p

    # Fallback: scan output dir for new files
    before = set(p.name for p in output_dir.rglob("*"))
    _ = _run_ytdlp(args + ["--print", "filename"])
    after = set(p.name for p in output_dir.rglob("*"))
    new = after - before
    for name in new:
        p = output_dir / name
        if p.exists() and p.stat().st_size > 1000:
            return p

    return None


def download_keyword(
    keyword: str,
    label: str,
    max_downloads: int | None = None,
) -> List[Path]:
    """Search and download videos for a keyword, return downloaded paths."""
    print(f"\n  Keyword: {keyword}")
    urls = search_videos(keyword, count=config.YTDLP_SEARCH_COUNT)
    limit = config.YTDLP_MAX_DOWNLOADS if max_downloads is None else max_downloads
    print(f"    Found {len(urls)} URLs, downloading up to {limit}")

    safe_kw = "".join(c if c.isalnum() or c in "-_" else "_" for c in keyword)[:40]
    out_dir = config.DOWNLOADS_DIR / label / safe_kw
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    for i, url in enumerate(urls):
        if len(downloaded) >= limit:
            break
        prefix = f"{label}_{i:03d}"
        path = download_video(url, out_dir, prefix)
        if path:
            downloaded.append(path)
            print(f"    [{len(downloaded)}/{min(len(urls), config.YTDLP_MAX_DOWNLOADS)}] {path.name}")
    return downloaded


def run_download_phase(max_per_keyword: int | None = None) -> dict:
    """Run the full download phase for all configured keywords.

    Returns dict mapping label → list of Paths.
    """
    all_downloaded: dict[str, list[Path]] = {}
    for label, keywords in config.KEYWORDS.items():
        print(f"\n{'='*60}")
        print(f"Class: {label}")
        print(f"{'='*60}")
        all_downloaded[label] = []
        for kw in keywords:
            paths = download_keyword(kw, label, max_downloads=max_per_keyword)
            all_downloaded[label].extend(paths)
    return all_downloaded
