#!/usr/bin/env python3
"""Import accepted Campus6 AI-screened RGB videos into the AVA review website.

This script preserves the existing human-label CSV.  Imported items are added
as a separate candidate batch with ``manual_label`` unset; their previous AI
label is shown in the page only as a suggestion, never written as human GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AUDIT_ROOT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1")
PRIMARY_ROOT = Path("/workspace/data/xzz_data/DAHUA/datasets/campus6_screened/accepted")
QWEN_MANIFEST = Path(
    "/workspace/data/xzz_data/DAHUA/datasets/"
    "campus6_screened_qwen_incremental/manifests/labeled_rgb.jsonl"
)
FFMPEG = Path("/workspace/code/envs/info_gcn/bin/ffmpeg")
FFPROBE = Path("/workspace/code/envs/info_gcn/bin/ffprobe")
BATCH = "ai_coarse_v1"
VIDEO_SUFFIXES = frozenset((".mp4", ".avi", ".mkv", ".webm", ".mov"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_part(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    ).strip("._") or "sample"


def stable_id(source: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{source}_{safe_part(path.stem)[:100]}_{digest}"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_records(primary_root: Path, qwen_manifest: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(primary_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            records.append(
                {
                    "source": "ai_primary",
                    "path": str(path.resolve()),
                    "ai_label": path.parent.name,
                    "source_label": path.parent.name,
                }
            )
    for item in read_jsonl(qwen_manifest):
        sample = item["sample"]
        final = item["final"]
        records.append(
            {
                "source": "ai_qwen_incremental",
                "path": str(Path(sample["path"]).resolve()),
                "ai_label": str(final["label"]),
                "source_label": str(sample.get("source_label", "")),
            }
        )
    unique: dict[str, dict[str, str]] = {}
    for record in records:
        if record["path"] in unique:
            raise ValueError(f"duplicate source path across accepted batches: {record['path']}")
        unique[record["path"]] = record
    missing = [record["path"] for record in records if not Path(record["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} accepted source videos are missing; first: {missing[0]}")
    return records


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def probe(path: Path) -> dict[str, str]:
    result = run(
        [
            str(FFPROBE), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ]
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    try:
        numerator, denominator = str(stream.get("r_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"no usable video stream after import: {path}")
    quality = "good"
    if duration < 2 or fps < 5 or width < 160 or height < 120:
        quality = "poor"
    elif duration < 5 or width < 320 or height < 240:
        quality = "usable"
    return {
        "duration": f"{duration:.3f}", "fps": f"{fps:.3f}",
        "width": str(width), "height": str(height),
        "frame_count": str(stream.get("nb_frames") or ""),
        "file_size": str(fmt.get("size") or path.stat().st_size),
        "technical_quality": quality,
    }


def materialize(source: Path, destination: Path) -> str:
    """Hard-link MP4s; transcode other browser-hostile formats to MP4."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return "existing"
    if source.suffix.lower() == ".mp4":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            os.symlink(source, destination)
            return "symlink"
    temporary = destination.with_suffix(".partial.mp4")
    temporary.unlink(missing_ok=True)
    result = run(
        [
            str(FFMPEG), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-map", "0:v:0", "-an", "-c:v", "libopenh264",
            "-b:v", "2M", "-movflags", "+faststart", str(temporary),
        ]
    )
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source}: {result.stderr[-800:]}")
    temporary.replace(destination)
    return "transcoded"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or ())


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_v3_module(script: Path):
    spec = importlib.util.spec_from_file_location("ava_probe_v3", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load website builder: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--qwen-manifest", type=Path, default=QWEN_MANIFEST)
    parser.add_argument("--batch", default=BATCH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit_root = args.audit_root.resolve()
    web = audit_root / "audit_gallery"
    manifests = audit_root / "candidate_manifests"
    existing_manifest = manifests / "probe_extraction_manifest_v3.csv"
    builder_script = audit_root / "scripts" / "probe_v3_completion_pipeline.py"
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("required info_gcn ffmpeg/ffprobe executables are unavailable")
    for path in (web / "index.html", existing_manifest, builder_script):
        if not path.is_file():
            raise FileNotFoundError(f"required AVA audit file is missing: {path}")

    records = source_records(args.primary_root.resolve(), args.qwen_manifest.resolve())
    by_source = {source: sum(row["source"] == source for row in records) for source in sorted({row["source"] for row in records})}
    suffixes = {suffix: sum(Path(row["path"]).suffix.lower() == suffix for row in records) for suffix in sorted({Path(row["path"]).suffix.lower() for row in records})}
    print(json.dumps({"batch": args.batch, "candidate_count": len(records), "by_source": by_source, "by_suffix": suffixes, "dry_run": args.dry_run}, ensure_ascii=False))
    if args.dry_run:
        return 0

    existing_rows, fields = read_csv(existing_manifest)
    required_fields = {"video_id", "candidate_group", "local_path", "decode_status"}
    if required_fields - set(fields):
        raise ValueError(f"existing manifest misses required columns: {sorted(required_fields - set(fields))}")
    old_batch_rows = [row for row in existing_rows if not row.get("candidate_group", "").startswith(args.batch + "_")]
    media_root = audit_root / "videos_probe" / args.batch
    batch_rows: list[dict[str, str]] = []
    ai_suggestions: list[dict[str, str]] = []
    action_counts: dict[str, int] = {}
    for index, record in enumerate(records, 1):
        source = Path(record["path"])
        group = f"{args.batch}_{record['source']}_{safe_part(record['ai_label'])}"
        identifier = stable_id(record["source"], source)
        destination = media_root / record["source"] / f"{identifier}.mp4"
        action = materialize(source, destination)
        action_counts[action] = action_counts.get(action, 0) + 1
        measured = probe(destination)
        row = {
            "video_id": identifier,
            "source": record["source"],
            "candidate_group": group,
            "kinetics_label": record["source_label"],
            "ava_actions": f"AI coarse suggestion: {record['ai_label']}",
            "mapping_status": "AI_COARSE_ACCEPTED",
            "shard": "",
            "extract_status": "imported",
            "decode_status": "ok",
            "local_path": str(destination),
            "duration": measured["duration"], "fps": measured["fps"],
            "width": measured["width"], "height": measured["height"],
            "frame_count": measured["frame_count"], "file_size": measured["file_size"],
            "technical_quality": measured["technical_quality"], "contact_sheet_path": "",
            "extract_error": "", "candidate_reason": (
                f"Imported {record['source']} AI-accepted candidate; "
                "AI suggestion is advisory only and requires human review."
            ),
            "same_scope": "ai_coarse_accepted", "timestamp": "", "bbox_person": "",
            "source_split": "",
        }
        batch_rows.append(row)
        ai_suggestions.append(
            {
                "video_id": identifier,
                "vlm_proposed_label": record["ai_label"],
                "vlm_confidence": "",
                "vlm_rationale": f"Imported AI coarse-screened result from {record['source']}; not human GT.",
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"processed {index}/{len(records)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = manifests / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_backup = backup_dir / f"probe_extraction_manifest_v3.before_{args.batch}_{timestamp}.csv"
    index_backup = backup_dir / f"audit_gallery_index.before_{args.batch}_{timestamp}.html"
    shutil.copy2(existing_manifest, manifest_backup)
    shutil.copy2(web / "index.html", index_backup)
    import_manifest = manifests / f"{args.batch}_candidates.csv"
    import_fields = list(batch_rows[0]) + ["original_source_path", "ai_suggestion", "imported_at"]
    import_rows = [
        {
            **row, "original_source_path": record["path"],
            "ai_suggestion": record["ai_label"], "imported_at": now(),
        }
        for row, record in zip(batch_rows, records)
    ]
    try:
        write_csv(existing_manifest, old_batch_rows + batch_rows, fields)
        write_csv(import_manifest, import_rows, import_fields)
        builder = load_v3_module(builder_script)
        builder.build_web(old_batch_rows + batch_rows, ai_suggestions)
        page = (web / "index.html").read_text(encoding="utf-8")
        # The V3 page intentionally shows only decode-valid rows.  Its manifest
        # also retains unavailable AVA/Kinetics targets, so counting every
        # manifest row would reject an otherwise valid candidate-pool publish.
        expected = sum(
            row.get("decode_status") == "ok"
            and row.get("split_status") != "split_parent"
            for row in (old_batch_rows + batch_rows)
        )
        if page.count("<section class='sample'") != expected:
            raise RuntimeError("generated page does not contain the expected number of samples")
    except Exception:
        shutil.copy2(manifest_backup, existing_manifest)
        shutil.copy2(index_backup, web / "index.html")
        raise
    print(json.dumps({
        "status": "complete", "candidate_count": len(batch_rows),
        "page_samples": expected,
        "media_actions": action_counts, "import_manifest": str(import_manifest),
        "manifest_backup": str(manifest_backup), "index_backup": str(index_backup),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
