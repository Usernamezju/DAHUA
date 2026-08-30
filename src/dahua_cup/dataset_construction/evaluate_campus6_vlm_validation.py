#!/usr/bin/env python3
"""Evaluate a fixed six-way VLM prompt on a frozen Campus6 validation split.

This utility deliberately does not alter annotations, model checkpoints, or
screening manifests.  It writes one append-only prediction record per sample,
so an interrupted API run can be resumed without reissuing completed calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dahua_cup.dataset_construction.video_mining import config
from dahua_cup.dataset_construction.video_mining.classify import (
    RateLimitExceeded,
    RateLimiter,
    classify_clip,
)


LABELS = tuple(config.CAMPUS6_CLASSES[:-1])
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PROMPT_VERSION = "campus6_fixed_six_way_v1"
FIXED_SIX_WAY_PROMPT = """You are evaluating a short real-world Campus6 behavior video.

Choose exactly one of these six labels from visible motion only. Do not use a
filename, subtitles, audio, metadata, or assumptions about the source:
1. normal_walk: ordinary walking, without pursuit or physical conflict.
2. normal_run: ordinary self-directed running, without pursuit.
3. playful_chase: two people chase playfully, with reciprocal, waiting, or
   game-like behavior and no clear fear or avoidance.
4. playful_push: two people playfully push or roughhouse; contact is light and
   both remain stable and continue friendly interaction.
5. conflict_chase: one person persistently chases another who visibly escapes,
   avoids, or accelerates defensively.
6. conflict_push: an aggressive one-sided push or shove; the target retreats,
   loses balance, or defends themselves.

When evidence is ambiguous, select the closest visible category but set
confidence to low. Output only this JSON object:
{"campus6_label":"one of the six labels above", "confidence":"high|medium|low", "reason":"brief visible evidence"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True,
                        help="Frozen ProtoGCN annotation pickle.")
    parser.add_argument("--pose-quality", type=Path, required=True,
                        help="JSONL mapping feature files to source videos.")
    parser.add_argument("--video-root", type=Path, required=True,
                        help="Root containing copied validation clips.")
    parser.add_argument("--source-video-root", type=Path,
                        help="Original video root when --video-root preserves source subdirectories.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Append-only JSONL predictions artifact.")
    parser.add_argument("--summary", type=Path, required=True,
                        help="Metrics JSON to overwrite from output records.")
    parser.add_argument("--api-key-file", type=Path, required=True,
                        help="Local secret file containing one DashScope API key.")
    parser.add_argument("--per-class", type=int, default=0,
                        help="Deterministically evaluate this many validation clips per class; 0 means all.")
    parser.add_argument("--max-requests", type=int, default=0,
                        help="Process at most this many pending rows before writing a resumable summary; 0 means all.")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def read_api_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value.rsplit(":", 1)[-1].strip()
    raise RuntimeError("API key file has no usable key")


def load_video_by_feature(path: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        feature = Path(row["feature"])
        videos[feature.stem] = Path(row["video"])
    return videos


def source_group(video: Path) -> str:
    stem = video.stem.rsplit("_clip", 1)[0]
    return f"{video.parent.name}/{stem}"


def sample_rows(annotation: Path, pose_quality: Path, per_class: int, seed: int) -> list[dict[str, Any]]:
    with annotation.open("rb") as stream:
        payload = pickle.load(stream)
    annotations = {row["frame_dir"]: row for row in payload["annotations"]}
    video_by_feature = load_video_by_feature(pose_quality)
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in payload["split"]["val"]:
        row = annotations[sample_id]
        label = LABELS[int(row["label"])]
        feature_stem = sample_id.rsplit("/", 1)[-1]
        video = video_by_feature.get(feature_stem)
        if video is None:
            raise RuntimeError(f"No source video mapping for {sample_id}")
        rows_by_label[label].append({
            "sample_id": sample_id,
            "label": label,
            "source_video": str(video),
            "source_group": source_group(video),
        })

    selected: list[dict[str, Any]] = []
    for label in LABELS:
        rows = rows_by_label[label]
        if not rows:
            raise RuntimeError(f"Validation split has no {label} samples")
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode("utf-8")
        ).hexdigest())
        if per_class:
            # Prefer distinct source videos before selecting another clip from one.
            unique, used_groups = [], set()
            for row in rows:
                if row["source_group"] not in used_groups:
                    unique.append(row)
                    used_groups.add(row["source_group"])
            rows = (unique + [row for row in rows if row not in unique])[:per_class]
        selected.extend(rows)
    return selected


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if (row.get("prompt_version") == PROMPT_VERSION
                    and row.get("prediction") in LABEL_TO_ID):
                completed[str(row["sample_id"])] = row
        except (TypeError, ValueError, KeyError):
            continue
    return completed


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    failed = 0
    for row in rows:
        predicted = row.get("prediction")
        if predicted not in LABEL_TO_ID:
            failed += 1
            continue
        matrix[LABEL_TO_ID[row["label"]]][LABEL_TO_ID[predicted]] += 1
    support = [sum(row) for row in matrix]
    correct = [matrix[index][index] for index in range(len(LABELS))]
    summary = {
        "prompt_version": PROMPT_VERSION,
        "evaluated": len(rows),
        "failed_requests": failed,
        "labels": list(LABELS),
        "confusion_matrix": matrix,
        "per_class_accuracy": {
            label: (correct[index] / support[index] if support[index] else None)
            for index, label in enumerate(LABELS)
        },
        "top1_accuracy": sum(correct) / sum(support) if sum(support) else None,
        "macro_accuracy": (
            sum(correct[index] / support[index] for index in range(len(LABELS)) if support[index])
            / sum(1 for count in support if count)
            if any(support) else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_clip_path(source: Path, label: str, video_root: Path,
                    source_video_root: Path | None) -> Path:
    if source_video_root is None:
        return video_root / label / source.name
    try:
        return video_root / source.relative_to(source_video_root)
    except ValueError as exc:
        raise ValueError(f"{source} is outside --source-video-root {source_video_root}") from exc


def main() -> None:
    args = parse_args()
    if args.per_class < 0:
        raise ValueError("--per-class must be non-negative")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative")
    if config.CLASSIFIER_BACKEND != "dashscope":
        raise RuntimeError("Set CAMPUS6_CLASSIFIER_BACKEND=dashscope for this evaluation")
    api_key = read_api_key(args.api_key_file)
    selected = sample_rows(args.annotation, args.pose_quality, args.per_class, args.seed)
    completed = load_completed(args.output)
    pending = [row for row in selected if row["sample_id"] not in completed]
    if args.max_requests:
        pending = pending[:args.max_requests]
    print(f"VLM validation: selected={len(selected)}, completed={len(completed)}, pending={len(pending)}")
    limiter = RateLimiter(config.DASHSCOPE_RPM)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for index, row in enumerate(pending, start=1):
            source = Path(row["source_video"])
            clip = local_clip_path(source, row["label"], args.video_root, args.source_video_root)
            if not clip.is_file():
                raise FileNotFoundError(f"Missing local validation clip: {clip}")
            try:
                limiter.wait()
                result = classify_clip(clip, api_key, prompt=FIXED_SIX_WAY_PROMPT)
            except RateLimitExceeded as exc:
                print(f"Provider quota reached after {index - 1} new requests: {exc}")
                break
            record = {
                **row,
                "prediction": result.get("campus6_label") if result else None,
                "confidence": result.get("confidence") if result else None,
                "reason": result.get("reason") if result else "classifier call failed",
                "prompt_version": PROMPT_VERSION,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            completed[record["sample_id"]] = record
            print(f"{index}/{len(pending)} {row['label']} -> {record['prediction']}")
    report_rows = [completed[row["sample_id"]] for row in selected if row["sample_id"] in completed]
    write_summary(report_rows, args.summary)


if __name__ == "__main__":
    main()
