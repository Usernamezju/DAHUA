#!/usr/bin/env python3
"""Split already-labelled Campus6 candidates into fixed-duration child clips.

The operation is intentionally conservative: it only selects a snapshot of
currently active manifest rows, copies their complete manual annotation to
each child, and marks the parent as ``split_parent``.  The original files and
rows are retained for traceability; the audit page excludes split parents.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probe(video: Path) -> dict[str, str]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration,size:stream=width,height,r_frame_rate", "-of", "json", str(video)],
        check=True, text=True, capture_output=True, timeout=60,
    )
    data = json.loads(result.stdout)
    stream = next((item for item in data.get("streams", []) if item.get("width")), {})
    duration = float(data["format"]["duration"])
    return {
        "duration": f"{duration:.3f}",
        "duration_seconds": f"{duration:.3f}",
        "file_size": str(data["format"].get("size", "")),
        "file_size_bytes": str(data["format"].get("size", "")),
        "width": str(stream.get("width", "")), "height": str(stream.get("height", "")),
        "fps": str(stream.get("r_frame_rate", "")),
    }


def encode_cut(source: Path, destination: Path, start: float, length: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-ss", f"{start:.3f}",
        "-t", f"{length:.3f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=300)
    if result.returncode or not destination.is_file() or destination.stat().st_size < 10_000:
        destination.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip()[-1000:] or "ffmpeg failed")
    codec = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(destination)],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if codec != "h264,yuv420p":
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"unexpected output encoding: {codec}")


def rebuild(pool: Path, manifest_rows: list[dict[str, str]], vlm_rows: list[dict[str, str]]) -> None:
    builder_path = pool / "scripts" / "probe_v3_completion_pipeline.py"
    spec = importlib.util.spec_from_file_location("campus6_reaudit_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.build_web(manifest_rows, vlm_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--label", default="aggressive_chasing")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--min-tail", type=float, default=2.5, help="Discard a shorter final remainder to keep clips comparable.")
    parser.add_argument("--repair-existing", action="store_true", help="Re-encode existing fixed children with a strict duration margin.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 <= args.min_tail <= args.seconds:
        raise ValueError("require seconds > 0 and 0 <= min-tail <= seconds")

    manifests = args.pool / "candidate_manifests"
    manifest_path = manifests / "probe_extraction_manifest_v3.csv"
    label_path = manifests / "manual_labels_v3.csv"
    vlm_path = manifests / "vlm_soft_review_v3.csv"
    order_path = manifests / "review_order_v3.csv"
    manifest_rows, manifest_fields = read_csv(manifest_path)
    label_rows, label_fields = read_csv(label_path)
    vlm_rows, vlm_fields = read_csv(vlm_path)
    order_rows, order_fields = read_csv(order_path)
    labels = {(row["video_id"], row["candidate_group"]): row for row in label_rows}
    if args.repair_existing:
        parents = {row.get("video_id", ""): row for row in manifest_rows}
        children = [row for row in manifest_rows if "__fixed_5000_" in row.get("video_id", "") and row.get("split_status") != "split_parent"]
        print(f"repair_targets={len(children)}")
        if args.dry_run:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = manifests / f"before_fixed_child_duration_repair_{stamp}"
        backup.mkdir()
        shutil.copy2(manifest_path, backup / manifest_path.name)
        for child in children:
            parent = parents.get(child.get("parent_video_id", ""))
            if parent is None:
                raise RuntimeError(f"parent not found for {child['video_id']}")
            source, target = Path(parent["local_path"]), Path(child["local_path"])
            start, end = float(child["segment_start"]), float(child["segment_end"])
            # Some frame rates make an ffmpeg '-t 5' container report slightly over 5s.
            # Reserve 0.1s for a full final frame, keeping every reported duration <= 5s.
            safe_length = min(end - start, args.seconds - 0.1)
            temporary = target.with_name(target.stem + ".repair.mp4")
            encode_cut(source, temporary, start, safe_length)
            temporary.replace(target)
            child["segment_end"] = f"{start + safe_length:.3f}"
            child.update(probe(target))
        write_csv(manifest_path, manifest_rows, manifest_fields)
        rebuild(args.pool, manifest_rows, vlm_rows)
        print(json.dumps({"backup": str(backup), "repaired": len(children)}, ensure_ascii=False))
        return
    targets = []
    for row in manifest_rows:
        label = labels.get((row.get("video_id", ""), row.get("candidate_group", "")), {})
        source = Path(row.get("local_path", ""))
        duration = float(row.get("duration") or row.get("duration_seconds") or 0)
        if label.get("manual_label") == args.label and row.get("split_status") != "split_parent" and duration > args.seconds + 0.02 and source.is_file():
            targets.append((row, label, source, duration))
    print(f"targets={len(targets)} label={args.label} segment_seconds={args.seconds}")
    for row, _, _, duration in targets:
        whole, tail = divmod(duration, args.seconds)
        count = int(whole) + (1 if tail >= args.min_tail else 0)
        note = "" if tail < args.min_tail else f" (last {tail:.3f}s)"
        print(f"  {row['video_id']}: {duration:.3f}s -> {count} clips{note}")
    if args.dry_run:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = manifests / f"before_{args.label}_{args.seconds:g}s_split_{stamp}"
    backup.mkdir()
    for path in (manifest_path, label_path, vlm_path, order_path):
        shutil.copy2(path, backup / path.name)

    existing_ids = {row.get("video_id", "") for row in manifest_rows}
    label_by_key = {(row.get("video_id", ""), row.get("candidate_group", "")): row for row in label_rows}
    rank = max((int(row.get("display_rank", "0") or 0) for row in order_rows), default=0)
    created = []
    for parent, parent_label, source, duration in targets:
        parent_id, group = parent["video_id"], parent["candidate_group"]
        parts = []
        start, index = 0.0, 1
        tail = duration % args.seconds
        usable_end = duration if tail >= args.min_tail else duration - tail
        while start < usable_end - 0.02:
            # Keep a 0.1s margin on nominal 5s clips: with certain source FPS,
            # the last complete frame otherwise makes the MP4 duration exceed 5s.
            nominal_width = min(args.seconds, usable_end - start)
            length = min(args.seconds - 0.1, nominal_width)
            child_id = f"{parent_id}__fixed_{int(args.seconds * 1000)}_{index:02d}"
            if child_id in existing_ids:
                raise RuntimeError(f"refusing to overwrite existing child: {child_id}")
            destination = source.parent / "review_splits" / f"{child_id}.mp4"
            encode_cut(source, destination, start, length)
            child = {field: "" for field in manifest_fields}
            child.update(parent)
            child.update({
                "video_id": child_id, "local_path": str(destination), "split_status": "", "parent_video_id": parent_id,
                "segment_start": f"{start:.3f}", "segment_end": f"{start + length:.3f}", "split_at": "", "split_end": "",
                "candidate_reason": f"Fixed {args.seconds:g}s child of labelled {parent_id}; inherited human annotation.",
                **probe(destination),
            })
            manifest_rows.append(child)
            child_label = {field: "" for field in label_fields}
            child_label.update(parent_label)
            child_label.update({"video_id": child_id, "candidate_group": group, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            label_rows.append(child_label)
            label_by_key[(child_id, group)] = child_label
            vrow = {field: "" for field in vlm_fields}
            vrow.update({"video_id": child_id, "candidate_group": group, "vlm_rationale": "Fixed-duration child; inherited verified human label.", "vlm_model": "human-label-inheritance"})
            vlm_rows.append(vrow)
            rank += 1
            order_rows.append({"video_id": child_id, "candidate_group": group, "display_rank": str(rank)})
            existing_ids.add(child_id)
            created.append((child_id, parent_label.get("manual_label", ""), length))
            parts.append(child_id)
            start += nominal_width
            index += 1
        parent["split_status"] = "split_parent"
        parent["split_at"] = f"{args.seconds:.3f}"
        parent["split_end"] = f"{duration:.3f}"
        parent["candidate_reason"] = (parent.get("candidate_reason", "") + f" | Replaced by {len(parts)} fixed <= {args.seconds:g}s labelled children.").strip()
    write_csv(manifest_path, manifest_rows, manifest_fields)
    write_csv(label_path, label_rows, label_fields)
    write_csv(vlm_path, vlm_rows, vlm_fields)
    write_csv(order_path, order_rows, order_fields)
    rebuild(args.pool, manifest_rows, vlm_rows)
    print(json.dumps({"backup": str(backup), "parents": len(targets), "children": len(created), "created": created}, ensure_ascii=False))


if __name__ == "__main__":
    main()
