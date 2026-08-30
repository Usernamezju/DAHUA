#!/usr/bin/env python3
"""Fast, resumable pose-extraction gate for Campus6 candidate videos.

The gate does not re-label actions.  It answers only whether a clip contains
enough complete human skeletons to remain a valid candidate.  It writes a JSONL
manifest instead of moving or deleting source videos, making the result auditable
and safe to apply later.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


INTERACTION_LABELS = {
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
}
ALL_LABELS = INTERACTION_LABELS | {"normal_walk", "normal_run"}
# COCO 17: nose, shoulders, hips, and ankles.  Requiring both ankles prevents
# first-person, heavily cropped, and upper-body-only clips entering Campus6.
COMPLETE_PERSON_JOINTS = (0, 5, 6, 11, 12, 15, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="screened/<label>/ tree")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO pose .pt weight")
    parser.add_argument("--output", type=Path, required=True, help="JSONL results manifest")
    parser.add_argument("--frames", type=int, default=8, help="uniform frames sampled per video")
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument("--joint-confidence", type=float, default=0.25)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', etc.; use cpu while GPU driver is unhealthy")
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated labels to process; empty means all six labels",
    )
    parser.add_argument("--limit", type=int, default=0, help="test with N unseen videos; 0 means all")
    return parser.parse_args()


def _label_for(path: Path, root: Path) -> str | None:
    try:
        label = path.relative_to(root).parts[0]
    except ValueError:
        return None
    return label if label in ALL_LABELS else None


def _videos(root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        label = _label_for(path, root)
        if label:
            yield label, path


def _sample_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        indices = sorted(set(np.linspace(0, total - 1, num=count, dtype=int).tolist()))
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def _complete_people(result, joint_confidence: float) -> int:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.conf is None:
        return 0
    confidence = keypoints.conf.detach().cpu().numpy()
    if confidence.ndim != 2 or confidence.shape[1] <= max(COMPLETE_PERSON_JOINTS):
        return 0
    complete = (confidence[:, COMPLETE_PERSON_JOINTS] >= joint_confidence).all(axis=1)
    return int(complete.sum())


def _load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if "video" in row and "pose_valid" in row:
                done.add(str(row["video"]))
        except (TypeError, ValueError):
            continue
    return done


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.min_valid_frames < 1:
        raise ValueError("--frames and --min-valid-frames must be positive")
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    from ultralytics import YOLO

    done = _load_done(args.output)
    selected_labels = {
        value.strip() for value in args.labels.split(",") if value.strip()
    }
    unknown_labels = selected_labels - ALL_LABELS
    if unknown_labels:
        raise ValueError(f"unknown --labels: {sorted(unknown_labels)}")
    pending = [
        (label, path)
        for label, path in _videos(args.root)
        if str(path.resolve()) not in done and (not selected_labels or label in selected_labels)
    ]
    if args.limit:
        pending = pending[: args.limit]
    print(f"Pose gate: {len(done)} completed, {len(pending)} pending", flush=True)

    model = YOLO(str(args.weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: Counter[str] = Counter()
    with args.output.open("a", encoding="utf-8") as stream:
        for index, (label, path) in enumerate(pending, 1):
            required_people = 2 if label in INTERACTION_LABELS else 1
            frames = _sample_frames(path, args.frames)
            complete_per_frame: list[int] = []
            error = ""
            try:
                if frames:
                    results = model(
                        frames,
                        conf=args.person_confidence,
                        imgsz=args.imgsz,
                        device=args.device,
                        verbose=False,
                    )
                    complete_per_frame = [
                        _complete_people(result, args.joint_confidence)
                        for result in results
                    ]
                else:
                    error = "decode_failed"
            except Exception as exc:  # preserve the candidate and record why
                error = f"pose_error:{type(exc).__name__}"
            valid_frames = sum(value >= required_people for value in complete_per_frame)
            pose_valid = bool(not error and valid_frames >= args.min_valid_frames)
            row = {
                "video": str(path.resolve()),
                "label": label,
                "pose_valid": pose_valid,
                "required_complete_people": required_people,
                "sampled_frames": len(frames),
                "valid_pose_frames": valid_frames,
                "complete_people_per_frame": complete_per_frame,
                "error": error,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            summary["accepted" if pose_valid else "rejected"] += 1
            if index % 25 == 0 or index == len(pending):
                print(
                    f"[{index}/{len(pending)}] accepted={summary['accepted']} "
                    f"rejected={summary['rejected']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
