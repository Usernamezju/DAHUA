#!/usr/bin/env python3
"""Review server with JSON label saving and lossless-parent video splitting."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/campus6_protogcn_reaudit_v1")
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
WEB = ROOT / "audit_gallery"
MANIFEST = ROOT / "candidate_manifests/probe_extraction_manifest_v3.csv"
VLM = ROOT / "candidate_manifests/vlm_soft_review_v3.csv"
ORDER = ROOT / "candidate_manifests/review_order_v3.csv"
LABEL = ROOT / "candidate_manifests/manual_labels_v3.csv"
BACKUP = ROOT / "candidate_manifests/manual_labels_v3.backup.csv"
LABEL_FIELDS = ["video_id", "candidate_group", "manual_label", "interaction_direction", "trajectory", "contact", "after_contact", "aggression_evidence", "playful_evidence", "scene", "pose_quality", "annotator_note", "vlm_decision", "updated_at"]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def probe(video: Path) -> dict[str, str]:
    result = subprocess.run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration,size:stream=width,height,r_frame_rate",
        "-of", "json", str(video),
    ], text=True, capture_output=True, check=True, timeout=60)
    payload = json.loads(result.stdout)
    stream = next((entry for entry in payload.get("streams", []) if entry.get("width")), {})
    duration = float(payload.get("format", {}).get("duration", 0))
    return {"duration_seconds": f"{duration:.3f}", "duration": f"{duration:.3f}", "file_size_bytes": str(payload.get("format", {}).get("size", "")), "width": str(stream.get("width", "")), "height": str(stream.get("height", "")), "fps": str(stream.get("r_frame_rate", ""))}


def cut(source: Path, target: Path, start: float, length: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{length:.3f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", str(target),
    ], text=True, capture_output=True, timeout=300)
    if result.returncode or not target.exists() or target.stat().st_size < 10_000:
        target.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip()[-500:] or "ffmpeg did not produce a usable clip")


def rebuild(manifest_rows: list[dict[str, str]], vlm_rows: list[dict[str, str]]) -> None:
    path = ROOT / "scripts/probe_v3_completion_pipeline.py"
    spec = importlib.util.spec_from_file_location("campus6_reaudit_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load audit-page builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.build_web(manifest_rows, vlm_rows)


def split_candidate(data: dict) -> dict:
    video_id, group = str(data.get("video_id", "")), str(data.get("candidate_group", ""))
    split_at = float(data.get("split_seconds", 0))
    rows, manifest_fields = read_csv(MANIFEST)
    vlm_rows, vlm_fields = read_csv(VLM)
    parent = next((row for row in rows if row.get("video_id") == video_id and row.get("candidate_group") == group), None)
    if parent is None:
        raise ValueError("candidate not found")
    if parent.get("split_status") == "split_parent":
        raise ValueError("this candidate has already been split")
    source = Path(parent.get("local_path", ""))
    if not source.is_file():
        raise ValueError("source video is unavailable")
    full_duration = float(probe(source)["duration_seconds"])
    if not (0.25 < split_at < full_duration - 0.25):
        raise ValueError("split time must leave both clips longer than 0.25 seconds")
    marker = int(round(split_at * 1000))
    child_dir = source.parent / "review_splits"
    first = child_dir / f"{video_id}__split_{marker}_part1.mp4"
    second = child_dir / f"{video_id}__split_{marker}_part2.mp4"
    cut(source, first, 0.0, split_at)
    try:
        cut(source, second, split_at, full_duration - split_at)
    except Exception:
        first.unlink(missing_ok=True)
        raise
    child_ids = [f"{video_id}__split_{marker}_part1", f"{video_id}__split_{marker}_part2"]
    if any(row.get("video_id") in child_ids for row in rows):
        raise ValueError("these split candidates already exist")
    parent["split_status"] = "split_parent"
    parent["split_at"] = f"{split_at:.3f}"
    parent["split_end"] = f"{full_duration:.3f}"
    parent["candidate_reason"] = (parent.get("candidate_reason", "") + " | Split into two review children.").strip()
    for child_id, child_file, start, end in zip(child_ids, (first, second), (0.0, split_at), (split_at, full_duration)):
        child = {field: "" for field in manifest_fields}
        child.update(parent)
        child.update({
            "video_id": child_id, "local_path": str(child_file), "split_status": "",
            "parent_video_id": video_id, "segment_start": f"{start:.3f}", "segment_end": f"{end:.3f}",
            "split_at": "", "split_end": "", "kinetics_label": "unlabeled",
            "ava_actions": "AI prediction: pending", "candidate_reason": f"Child of {video_id}, manually split at {split_at:.3f}s; manual review required.",
            **probe(child_file),
        })
        rows.append(child)
        vrow = {field: "" for field in vlm_fields}
        vrow.update({"video_id": child_id, "vlm_rationale": "Manual split child; no AI label assigned.", "vlm_model": ""})
        vlm_rows.append(vrow)
    write_csv(MANIFEST, rows, manifest_fields)
    write_csv(VLM, vlm_rows, vlm_fields)
    order_rows, order_fields = read_csv(ORDER)
    rank = max((int(row.get("display_rank", "0") or 0) for row in order_rows), default=0)
    for child_id in child_ids:
        rank += 1
        order_rows.append({"video_id": child_id, "candidate_group": group, "display_rank": str(rank)})
    write_csv(ORDER, order_rows, order_fields or ["video_id", "candidate_group", "display_rank"])
    rebuild(rows, vlm_rows)
    return {"ok": True, "parent": video_id, "children": child_ids}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/labels_v3":
            rows, _ = read_csv(LABEL)
            self.send_json(200, {row["video_id"] + "|" + row["candidate_group"]: row for row in rows})
            return
        super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/save_v3":
                rows, _ = read_csv(LABEL)
                if LABEL.exists():
                    shutil.copy2(LABEL, BACKUP)
                data["updated_at"] = now()
                key = (data.get("video_id", ""), data.get("candidate_group", ""))
                replaced = False
                for index, row in enumerate(rows):
                    if (row.get("video_id", ""), row.get("candidate_group", "")) == key:
                        rows[index] = {**row, **data}
                        replaced = True
                        break
                if not replaced:
                    rows.append(data)
                write_csv(LABEL, rows, LABEL_FIELDS)
                self.send_json(200, {"ok": True})
            elif self.path == "/split_v3":
                self.send_json(200, split_candidate(data))
            else:
                self.send_json(404, {"ok": False, "error": "unknown endpoint"})
        except Exception as exc:
            self.send_json(400, {"ok": False, "error": str(exc)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7873)
    arguments = parser.parse_args()
    print(f"Listening on http://127.0.0.1:{arguments.port}/")
    ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler).serve_forever()
