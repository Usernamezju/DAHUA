"""Export frozen-YOLO person tracks for Group STN distillation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dahua_cup.pipeline.common import log_event, read_jsonl, write_jsonl
from dahua_cup.stn.teacher_labels import (
    SCHEMA_VERSION,
    crowd_frame_ratio,
    interpolate_track_gaps,
    normalize_xyxy,
    select_track_ids,
    validate_teacher_record,
)


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
RUN_SCHEMA_VERSION = "stn_yolo_export.v2"
BEHAVE_SEGMENTS = (
    (1, 11200),
    (11500, 17450),
    (18000, 23700),
    (24300, 35200),
    (35450, 47160),
    (47300, 58400),
    (59800, 66750),
    (67210, 76800),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-manifest",
        help="CSV containing clip_id/path, such as manual_label_manifest.csv",
    )
    source.add_argument("--video-root", help="Recursively scan videos")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("DAHUA_STN_YOLO_TEACHER"),
        help="Local Ultralytics YOLO detector/pose weight",
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--device",
        default="0",
        help="One Ultralytics device, such as 0 or cpu",
    )
    device.add_argument(
        "--devices",
        help="Comma-separated physical GPU indices for video-level parallelism",
    )
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--minimum-track-coverage", type=float, default=0.20)
    parser.add_argument("--maximum-interpolation-gap", type=int, default=3)
    parser.add_argument("--maximum-crowd-ratio", type=float, default=0.10)
    parser.add_argument(
        "--crowd-policy",
        choices=("skip", "top2"),
        default="skip",
        help="Skip persistently crowded clips or retain two strongest tracks",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument(
        "--finalize-partial",
        action="store_true",
        help=(
            "Merge currently accepted part records without running pending "
            "videos; preserve pending videos as unprocessed"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_split_group(
    path: Path, dataset: str, sample_id: str
) -> str:
    dataset_upper = dataset.upper()
    if dataset_upper == "KTH":
        match = re.search(r"(person\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            return f"kth_{match.group(1).lower()}"
    if dataset_upper == "BEHAVE":
        numbers = [int(value) for value in re.findall(r"\d+", path.stem)]
        if numbers:
            start = numbers[-2] if len(numbers) >= 2 else numbers[-1]
            for segment_start, segment_end in BEHAVE_SEGMENTS:
                if segment_start <= start <= segment_end:
                    return f"behave_{segment_start}_{segment_end}"
    if dataset_upper == "LIMU":
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return f"limu_pair_{suffix}"
    return sample_id


def discover_videos(args) -> list[dict]:
    rows = []
    if args.input_manifest:
        source = Path(args.input_manifest).expanduser().resolve()
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                raw_path = row.get("path") or row.get("video_path")
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = source.parent / path
                path = path.resolve()
                if path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                sample_id = str(
                    row.get("clip_id") or row.get("sample_id") or path.stem
                )
                dataset = str(row.get("source_dataset") or "")
                rows.append({
                    "sample_id": sample_id,
                    "video_path": path,
                    "source_dataset": dataset,
                    "source_label": str(row.get("source_label") or ""),
                    "split_group": str(row.get("split_group") or "")
                    or _default_split_group(path, dataset, sample_id),
                })
    else:
        root = Path(args.video_root).expanduser().resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            dataset = path.parent.name
            rows.append({
                "sample_id": path.stem,
                "video_path": path,
                "source_dataset": dataset,
                "source_label": "",
                "split_group": _default_split_group(
                    path, dataset, path.stem
                ),
            })
    unique = {}
    for row in rows:
        key = str(row["video_path"])
        unique.setdefault(key, row)
    values = list(unique.values())
    if args.max_videos > 0:
        values = values[: args.max_videos]
    return values


def _fallback_track_ids(
    boxes: list[list[float]],
    previous: dict[int, list[float]],
    next_track_id: int,
) -> tuple[list[int], dict[int, list[float]], int]:
    """Small deterministic IoU tracker used only when Ultralytics has no IDs."""

    def iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / max(
            first_area + second_area - intersection, 1e-8
        )

    assigned = []
    available = set(previous)
    current = {}
    for box in boxes:
        candidates = sorted(
            ((iou(box, previous[value]), value) for value in available),
            reverse=True,
        )
        if candidates and candidates[0][0] >= 0.30:
            track_id = candidates[0][1]
            available.remove(track_id)
        else:
            track_id = next_track_id
            next_track_id += 1
        assigned.append(track_id)
        current[track_id] = box
    return assigned, current, next_track_id


def infer_record(teacher, source: dict, args) -> tuple[dict | None, str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("YOLO teacher export requires opencv-python") from exc

    video_path = Path(source["video_path"])
    if not video_path.is_file():
        return None, "video_missing"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, "video_unreadable"
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        return None, "invalid_video_metadata"

    results = teacher.track(
        source=str(video_path),
        stream=True,
        persist=False,
        tracker=args.tracker,
        classes=[0],
        conf=args.confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )
    frames = []
    previous_boxes = {}
    next_track_id = 1_000_000
    for frame_index, result in enumerate(results):
        persons = []
        if result.boxes is not None and len(result.boxes):
            raw_boxes = result.boxes.xyxy.detach().cpu().tolist()
            raw_scores = result.boxes.conf.detach().cpu().tolist()
            ids = (
                result.boxes.id.detach().cpu().tolist()
                if result.boxes.id is not None else None
            )
            normalized = [
                normalize_xyxy(box, width, height) for box in raw_boxes
            ]
            if ids is None:
                ids, previous_boxes, next_track_id = _fallback_track_ids(
                    normalized, previous_boxes, next_track_id
                )
            else:
                ids = [int(value) for value in ids]
                previous_boxes = dict(zip(ids, normalized))
            persons = [
                {
                    "track_id": int(track_id),
                    "bbox_xyxy": box,
                    "confidence": float(score),
                    "interpolated": False,
                }
                for track_id, box, score in zip(
                    ids, normalized, raw_scores
                )
            ]
        else:
            previous_boxes = {}
        frames.append({"frame_index": frame_index, "persons": persons})
    if not frames:
        return None, "no_decoded_frames"
    crowd_ratio = crowd_frame_ratio(frames)
    if (
        args.crowd_policy == "skip"
        and crowd_ratio > args.maximum_crowd_ratio
    ):
        return None, "crowded_clip"
    selected = select_track_ids(
        frames,
        maximum_people=2,
        minimum_coverage=args.minimum_track_coverage,
    )
    if not selected:
        return None, "no_persistent_person"
    selected_frames = interpolate_track_gaps(
        frames,
        selected,
        maximum_gap=args.maximum_interpolation_gap,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": source["sample_id"],
        "video_path": str(video_path.resolve()),
        "source_dataset": source.get("source_dataset", ""),
        "source_label": source.get("source_label", ""),
        "split_group": source.get("split_group", source["sample_id"]),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": len(frames),
        "selected_track_ids": selected,
        "crowd_frame_ratio": crowd_ratio,
        "teacher": {
            "framework": "ultralytics",
            "model": str(args.model),
            "confidence_threshold": args.confidence,
            "image_size": args.image_size,
            "tracker": args.tracker,
        },
        "frames": selected_frames,
    }
    validate_teacher_record(record)
    return record, "accepted"


def parse_devices(value: str) -> list[int]:
    devices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            device = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid GPU index: {part}") from exc
        if device < 0:
            raise ValueError("GPU indices must be non-negative")
        if device in devices:
            raise ValueError(f"duplicate GPU index: {device}")
        devices.append(device)
    if not devices:
        raise ValueError("--devices must contain at least one GPU index")
    return devices


def shard_sources(sources: list[dict], shard_count: int) -> list[list[dict]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards = [[] for _ in range(shard_count)]
    for index, source in enumerate(sources):
        shards[index % shard_count].append(source)
    return shards


def _source_key(source: dict) -> str:
    return str(Path(source["video_path"]).resolve())


def _source_digest(sources: list[dict]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        value = {
            "sample_id": source["sample_id"],
            "video_path": _source_key(source),
            "source_dataset": source.get("source_dataset", ""),
            "source_label": source.get("source_label", ""),
            "split_group": source.get("split_group", ""),
        }
        digest.update(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _semantic_settings(args, sources: list[dict]) -> dict:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "input_manifest": (
            str(Path(args.input_manifest).resolve())
            if args.input_manifest
            else None
        ),
        "video_root": (
            str(Path(args.video_root).resolve()) if args.video_root else None
        ),
        "source_count": len(sources),
        "source_digest": _source_digest(sources),
        "model": str(args.model),
        "tracker": args.tracker,
        "image_size": args.image_size,
        "confidence": args.confidence,
        "minimum_track_coverage": args.minimum_track_coverage,
        "maximum_interpolation_gap": args.maximum_interpolation_gap,
        "maximum_crowd_ratio": args.maximum_crowd_ratio,
        "crowd_policy": args.crowd_policy,
    }


def _write_metadata(parts_dir: Path, settings: dict) -> None:
    path = parts_dir / "run.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != settings:
            raise RuntimeError(
                f"{parts_dir} belongs to a different export configuration; "
                "use a different --output path"
            )
        return
    parts_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _load_progress(parts_dir: Path) -> dict[str, dict]:
    processed = {}
    for path in sorted(parts_dir.glob("progress-*.jsonl")):
        for row in read_jsonl(path):
            processed[str(row["video_path"])] = row
    return processed


def merge_part_records(
    part_paths: list[Path], sources: list[dict]
) -> list[dict]:
    by_path = {}
    for path in sorted(part_paths):
        if not path.is_file():
            continue
        for record in read_jsonl(path):
            validate_teacher_record(record)
            key = str(Path(record["video_path"]).resolve())
            by_path[key] = record
    return [
        by_path[key]
        for key in (_source_key(source) for source in sources)
        if key in by_path
    ]


def _export_summary(
    sources: list[dict],
    progress: dict[str, dict],
    accepted: list[dict],
    devices: list,
    output: Path,
    parts_dir: Path,
    *,
    partial: bool,
) -> dict:
    rejected = {}
    for row in progress.values():
        if row["status"] == "accepted":
            continue
        reason = str(row["status"])
        rejected[reason] = rejected.get(reason, 0) + 1
    completed = set(progress)
    completed.update(
        str(Path(record["video_path"]).resolve()) for record in accepted
    )
    source_keys = {_source_key(source) for source in sources}
    return {
        "schema_version": SCHEMA_VERSION,
        "input_videos": len(sources),
        "processed": len(source_keys & completed),
        "accepted": len(accepted),
        "rejected": rejected,
        "unprocessed": len(source_keys - completed),
        "partial": partial,
        "devices": devices,
        "output": str(output),
        "parts_dir": str(parts_dir),
    }


def _write_summary(output: Path, summary: dict) -> Path:
    path = Path(str(output) + ".summary.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _worker_name(worker_index: int, device) -> str:
    safe_device = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(device))
    return f"worker-{worker_index:02d}-device-{safe_device}"


def _worker(
    worker_index: int,
    device,
    sources: list[dict],
    options: dict,
    parts_dir: str,
    overall_total: int,
) -> None:
    from ultralytics import YOLO

    worker_name = _worker_name(worker_index, device)
    accepted_path = Path(parts_dir) / f"accepted-{worker_name}.jsonl"
    progress_path = Path(parts_dir) / f"progress-{worker_name}.jsonl"
    args = SimpleNamespace(**options, device=str(device))
    teacher = YOLO(args.model)
    for shard_index, source in enumerate(sources, 1):
        error = None
        try:
            record, reason = infer_record(teacher, source, args)
        except Exception as exc:  # preserve the rest of a long export
            record = None
            reason = "inference_error"
            error = f"{type(exc).__name__}: {exc}"
        if record is not None:
            _append_jsonl(accepted_path, record)
        progress = {
            "sample_id": source["sample_id"],
            "video_path": _source_key(source),
            "status": reason,
            "device": str(device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            progress["error"] = error
        _append_jsonl(progress_path, progress)
        log_event(
            "stn_yolo_teacher_progress",
            device=str(device),
            shard_current=shard_index,
            shard_total=len(sources),
            overall_total=overall_total,
            sample_id=source["sample_id"],
            status=reason,
        )


def _validate_args(args) -> None:
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be in [0,1]")
    if not 0 <= args.minimum_track_coverage <= 1:
        raise ValueError("--minimum-track-coverage must be in [0,1]")
    if not 0 <= args.maximum_crowd_ratio <= 1:
        raise ValueError("--maximum-crowd-ratio must be in [0,1]")
    if args.image_size < 32 or args.maximum_interpolation_gap < 0:
        raise ValueError("invalid image size or interpolation gap")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    devices = (
        parse_devices(args.devices) if args.devices else [args.device]
    )
    sources = discover_videos(args)
    if not sources:
        raise RuntimeError("no input videos were discovered")
    preview_shards = shard_sources(sources, len(devices))
    if args.dry_run:
        print(json.dumps(
            {
                "input_videos": len(sources),
                "devices": devices,
                "videos_per_device": {
                    str(device): len(shard)
                    for device, shard in zip(devices, preview_shards)
                },
                "output": str(Path(args.output).resolve()),
                "model": args.model,
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
        return
    if not args.model:
        raise ValueError("--model or DAHUA_STN_YOLO_TEACHER is required")

    output = Path(args.output).expanduser().resolve()
    parts_dir = Path(str(output) + ".parts")
    _write_metadata(parts_dir, _semantic_settings(args, sources))
    processed = _load_progress(parts_dir)
    pending = [
        source for source in sources if _source_key(source) not in processed
    ]
    if args.finalize_partial:
        accepted = merge_part_records(
            list(parts_dir.glob("accepted-*.jsonl")), sources
        )
        if not accepted:
            raise RuntimeError(
                "no accepted part records are available to finalize"
            )
        write_jsonl(output, accepted)
        summary = _export_summary(
            sources,
            processed,
            accepted,
            devices,
            output,
            parts_dir,
            partial=bool(pending),
        )
        summary["summary"] = str(_write_summary(output, summary))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return
    shards = shard_sources(pending, len(devices))
    options = {
        "model": args.model,
        "tracker": args.tracker,
        "image_size": args.image_size,
        "confidence": args.confidence,
        "minimum_track_coverage": args.minimum_track_coverage,
        "maximum_interpolation_gap": args.maximum_interpolation_gap,
        "maximum_crowd_ratio": args.maximum_crowd_ratio,
        "crowd_policy": args.crowd_policy,
    }
    context = multiprocessing.get_context("spawn")
    processes = []
    for worker_index, (device, shard) in enumerate(
        zip(devices, shards)
    ):
        if not shard:
            continue
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                device,
                shard,
                options,
                str(parts_dir),
                len(sources),
            ),
            name=f"stn-yolo-{device}",
        )
        process.start()
        processes.append(process)
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    failed = [
        f"{process.name}:{process.exitcode}"
        for process in processes
        if process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(
            "YOLO workers failed; partial records are resumable: "
            + ", ".join(failed)
        )

    progress = _load_progress(parts_dir)
    accepted = merge_part_records(
        list(parts_dir.glob("accepted-*.jsonl")), sources
    )
    if not accepted:
        raise RuntimeError("YOLO did not produce any usable STN records")
    write_jsonl(output, accepted)
    summary = _export_summary(
        sources,
        progress,
        accepted,
        devices,
        output,
        parts_dir,
        partial=False,
    )
    summary["summary"] = str(_write_summary(output, summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
