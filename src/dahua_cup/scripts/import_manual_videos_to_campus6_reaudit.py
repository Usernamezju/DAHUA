#!/usr/bin/env python3
"""Append locally supplied videos to the Campus6 re-annotation webpage.

The script only adds previously unseen files, preserves all submitted manual
reviews, and keeps these clips separate from train/eval data via a dedicated
candidate group.  It deliberately creates no AI label for them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


POOL_DEFAULT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/campus6_protogcn_reaudit_v1")
GROUP = "manual_conflict_chase_upload"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probe(video: Path) -> dict[str, str]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=width,height,r_frame_rate",
        "-of", "json", str(video),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("width")), {})
    return {
        "duration_seconds": f"{float(payload.get('format', {}).get('duration', 0)):.3f}",
        "file_size_bytes": str(payload.get("format", {}).get("size", "")),
        "width": str(stream.get("width", "")), "height": str(stream.get("height", "")),
        "fps": str(stream.get("r_frame_rate", "")),
    }


def build_page(pool: Path, manifest_rows: list[dict[str, str]], vlm_rows: list[dict[str, str]]) -> None:
    builder_path = pool / "scripts/probe_v3_completion_pipeline.py"
    spec = importlib.util.spec_from_file_location("campus6_manual_import_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review builder: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.build_web(manifest_rows, vlm_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=POOL_DEFAULT)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    pool, source_dir = args.pool.resolve(), args.source_dir.resolve()
    manifests = pool / "candidate_manifests"
    manifest_path = manifests / "probe_extraction_manifest_v3.csv"
    vlm_path = manifests / "vlm_soft_review_v3.csv"
    rows, vlm_rows = read_csv(manifest_path), read_csv(vlm_path)
    manifest_fields = list(rows[0])
    vlm_fields = list(vlm_rows[0])
    existing_by_path = {row.get("local_path", "") for row in rows}
    existing_ids = {row.get("video_id", "") for row in rows}
    added: list[str] = []

    for video in sorted(source_dir.rglob("*")):
        if video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"} or not video.is_file():
            continue
        video = video.resolve()
        if str(video) in existing_by_path:
            continue
        digest = hashlib.sha1(video.read_bytes()).hexdigest()[:16]
        video_id = f"manual_conflict_chase_{digest}"
        if video_id in existing_ids:
            continue
        try:
            measured = probe(video)
        except Exception as exc:
            print(f"skip unreadable {video}: {exc}", file=sys.stderr)
            continue
        row = {field: "" for field in manifest_fields}
        row.update({
            "video_id": video_id, "source": "Desktop_manual_upload",
            "candidate_group": GROUP, "kinetics_label": "unlabeled",
            "ava_actions": "AI prediction: pending", "mapping_status": "MANUAL_REVIEW_CANDIDATE",
            "extract_status": "imported", "decode_status": "ok", "local_path": str(video),
            "candidate_reason": "User-supplied conflict-chase candidate; manual label required.",
            "same_scope": "campus6_protogcn_reaudit_v1", "source_split": "manual_upload",
            **measured,
        })
        rows.append(row)
        vrow = {field: "" for field in vlm_fields}
        vrow.update({
            "video_id": video_id, "vlm_rationale": "No AI prediction: user-supplied clip awaiting manual review.",
            "vlm_model": "", "vlm_proposed_label": "", "vlm_confidence": "",
        })
        vlm_rows.append(vrow)
        existing_ids.add(video_id)
        existing_by_path.add(str(video))
        added.append(video_id)

    write_csv(manifest_path, rows, manifest_fields)
    write_csv(vlm_path, vlm_rows, vlm_fields)
    order_path = manifests / "review_order_v3.csv"
    order = read_csv(order_path) if order_path.exists() else []
    known_order = {row.get("video_id", "") for row in order}
    rank = max((int(row.get("display_rank", "0") or 0) for row in order), default=0)
    for row in rows:
        if row["video_id"] not in known_order:
            rank += 1
            order.append({"video_id": row["video_id"], "candidate_group": row["candidate_group"], "display_rank": str(rank)})
    write_csv(order_path, order, ["video_id", "candidate_group", "display_rank"])
    build_page(pool, rows, vlm_rows)
    print(json.dumps({"added": len(added), "total": len(rows), "group": GROUP, "video_ids": added}, ensure_ascii=False))


if __name__ == "__main__":
    main()
