#!/usr/bin/env python3
"""Repair browser-hostile video codecs used by the AVA review website.

Only media files referenced by decode-valid rows in the existing V3 manifest
are considered.  The script leaves manifests and human annotations unchanged.
Each replacement is validated as H.264, then atomically swapped in place after
the original has been copied to a timestamped backup directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


AUDIT_ROOT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1")
FFMPEG = Path("/workspace/code/envs/info_gcn/bin/ffmpeg")
FFPROBE = Path("/workspace/code/envs/info_gcn/bin/ffprobe")
# H.264 MP4 is the common denominator for Chromium/Edge/Firefox playback.
BROWSER_HOSTILE_CODECS = frozenset(
    {"mpeg4", "hevc", "h265", "mpeg2video", "msmpeg4v3", "wmv3", "mjpeg"}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def probe(path: Path) -> dict[str, str]:
    result = run(
        [
            str(FFPROBE), "-v", "error", "-show_entries",
            "stream=index,codec_type,codec_name,pix_fmt,width,height,duration",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ]
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-800:] or "ffprobe returned a non-zero exit status")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("no video stream found")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}
    return {
        "video_codec": str(video.get("codec_name") or ""),
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "width": str(video.get("width") or ""),
        "height": str(video.get("height") or ""),
        "duration": str(video.get("duration") or fmt.get("duration") or ""),
        "audio_codec": str((audio or {}).get("codec_name") or ""),
        "size": str(fmt.get("size") or path.stat().st_size),
    }


def decode_validate(path: Path) -> None:
    result = run([str(FFMPEG), "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if result.returncode:
        raise RuntimeError(result.stderr[-800:] or "ffmpeg decode validation failed")


def manifest_paths(manifest: Path, audit_root: Path) -> list[Path]:
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths: list[Path] = []
    seen: set[Path] = set()
    media_root = (audit_root / "videos_probe").resolve()
    for row in rows:
        if row.get("decode_status") != "ok":
            continue
        path = Path(row.get("local_path", "")).resolve()
        if path in seen:
            continue
        if not path.is_relative_to(media_root):
            raise ValueError(f"manifest path outside videos_probe: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest media file is missing: {path}")
        seen.add(path)
        paths.append(path)
    return paths


def repair(path: Path, audit_root: Path, backup_root: Path) -> dict[str, str]:
    before = probe(path)
    if before["video_codec"] not in BROWSER_HOSTILE_CODECS:
        return {"path": str(path), "action": "unchanged", **before}

    temporary = path.with_name(f"{path.stem}.webfix.partial.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(FFMPEG), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libopenh264",
        "-b:v", "2M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(temporary),
    ]
    result = run(command)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[-800:]}")
    try:
        after = probe(temporary)
        if after["video_codec"] != "h264":
            raise RuntimeError(f"unexpected output video codec: {after['video_codec']}")
        decode_validate(temporary)
        backup = backup_root / path.relative_to(audit_root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path), "action": "reencoded_h264", "backup": str(backup), **after}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit_root = args.audit_root.resolve()
    manifest = audit_root / "candidate_manifests" / "probe_extraction_manifest_v3.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("required FFmpeg/FFprobe binaries are unavailable")

    paths = manifest_paths(manifest, audit_root)
    audit: list[dict[str, str]] = []
    for path in paths:
        details = probe(path)
        audit.append({"path": str(path), "action": "audit", **details})
    codec_counts = Counter(item["video_codec"] for item in audit)
    targets = [item for item in audit if item["video_codec"] in BROWSER_HOSTILE_CODECS]
    summary = {
        "timestamp": utc_now(),
        "manifest": str(manifest),
        "decode_ok_media": len(paths),
        "video_codec_counts": dict(sorted(codec_counts.items())),
        "browser_hostile_targets": len(targets),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in targets:
        print(f"TARGET {item['video_codec']} {item['path']}")
    if args.dry_run:
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = audit_root / "videos_probe" / "_codec_backups" / run_id
    report_path = audit_root / "reports" / f"WEB_VIDEO_CODEC_REPAIR_{run_id}.json"
    repaired: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for index, item in enumerate(targets, 1):
        path = Path(item["path"])
        try:
            repaired.append(repair(path, audit_root, backup_root))
            print(f"repaired {index}/{len(targets)}: {path.name}", flush=True)
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
            print(f"failed {index}/{len(targets)}: {path.name}: {error}", file=sys.stderr, flush=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                **summary,
                "dry_run": False,
                "backup_root": str(backup_root),
                "repaired": repaired,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "repaired": len(repaired), "failures": len(failures)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
