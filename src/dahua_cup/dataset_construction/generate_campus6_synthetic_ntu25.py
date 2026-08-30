#!/usr/bin/env python3
"""Create controlled NTU25 skeleton augmentation from Campus6 train poses only.

The generator retains real per-joint body templates, then resamples, rotates,
and places the two actors on class-specific root trajectories.  It is intended
only for training augmentation: the input validation and test splits are copied
verbatim into the output annotation file.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)
INTERACTION_IDS = {2, 3, 4, 5}
DEFAULT_COUNTS = "normal_walk=0,normal_run=200,playful_chase=200,playful_push=200,conflict_chase=200,conflict_push=200"


@dataclass(frozen=True)
class MotionControl:
    frames: int
    yaw: float
    scale: float
    speed: float
    separation: float
    noise_std: float


def parse_counts(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    names = {name: index for index, name in enumerate(LABELS)}
    for token in value.split(","):
        name, separator, raw_count = token.strip().partition("=")
        if not separator or name not in names:
            raise ValueError(f"Invalid --counts term: {token!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError("Synthetic counts must be non-negative")
        result[names[name]] = count
    missing = set(range(len(LABELS))) - set(result)
    if missing:
        raise ValueError("--counts must include every class")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-annotation", type=Path, required=True)
    parser.add_argument("--output-annotation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--counts", default=DEFAULT_COUNTS)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--noise-std", type=float, default=0.006)
    parser.add_argument("--yaw-radians", type=float, default=math.pi)
    parser.add_argument("--max-attempts", type=int, default=20)
    return parser.parse_args()


def resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    """Linearly resample [M,T,V,C] without relying on video frame rate."""
    old_frames = sequence.shape[1]
    if old_frames == frames:
        return sequence.copy()
    positions = np.linspace(0, old_frames - 1, frames, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, old_frames - 1)
    alpha = (positions - lower).reshape(1, frames, 1, 1)
    return (sequence[:, lower] * (1.0 - alpha) + sequence[:, upper] * alpha).astype(np.float32)


def rotate_z(points: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated = points.copy()
    x, y = points[..., 0], points[..., 1]
    rotated[..., 0] = cosine * x - sine * y
    rotated[..., 1] = sine * x + cosine * y
    return rotated


def valid_person(keypoint: np.ndarray, person: int) -> bool:
    values = keypoint[person]
    # MediaPipe clips can contain a long zero-padded tail.  A body is usable
    # when a meaningful portion has spatially spread joints, not only when
    # every temporal slot is non-zero.
    present = np.linalg.norm(values, axis=-1) > 1e-5
    return bool(
        np.isfinite(values).all()
        and present.mean() >= 0.15
        and float(values[present].std()) > 0.02
    )


def local_templates(keypoint: np.ndarray, frames: int) -> np.ndarray:
    sampled = resample(keypoint, frames)
    if not valid_person(sampled, 0):
        raise ValueError("Reference has no valid primary person")
    roots = sampled[:, :, :1, :]
    local = sampled - roots
    if not valid_person(sampled, 1):
        # A reflected copy still supplies a coherent second body for interaction
        # controls when MediaPipe tracked only the primary actor.
        local[1] = local[0]
        local[1, :, :, 0] *= -1.0
    return local


def root_trajectories(label: int, control: MotionControl) -> np.ndarray:
    """Return [2,T,3] actor roots; x is travel, y is lateral separation."""
    phase = np.linspace(0.0, 1.0, control.frames, dtype=np.float32)
    roots = np.zeros((2, control.frames, 3), dtype=np.float32)
    sway = 0.04 * np.sin(2.0 * np.pi * (phase + 0.15))
    if label in (0, 1):
        roots[0, :, 0] = control.speed * (phase - 0.5)
        roots[0, :, 1] = sway
        roots[1] = roots[0]
        roots[1, :, 1] += 3.5  # Keep the inactive template outside interaction range.
    elif label == 2:  # playful chase: close, wavy, with a role-like crossover.
        lead = control.speed * (phase - 0.5)
        distance = control.separation + 0.10 * np.sin(2.0 * np.pi * phase)
        roots[0, :, 0] = lead - distance * np.cos(np.pi * phase)
        roots[1, :, 0] = lead + distance * np.cos(np.pi * phase)
        roots[0, :, 1] = 0.20 + 0.10 * np.sin(2.0 * np.pi * phase)
        roots[1, :, 1] = -0.20 - 0.10 * np.sin(2.0 * np.pi * phase)
    elif label == 4:  # conflict chase: target keeps a unidirectional lead.
        target = control.speed * (phase - 0.5)
        distance = control.separation * (1.25 - 0.45 * phase)
        roots[0, :, 0] = target - distance
        roots[1, :, 0] = target
        roots[0, :, 1] = 0.04 * np.sin(4.0 * np.pi * phase)
        roots[1, :, 1] = -roots[0, :, 1]
    else:
        # Both push variants start face-to-face.  The second actor translates at
        # contact; conflict uses a larger, abrupt retreat than playful pushing.
        contact = 1.0 / (1.0 + np.exp(-20.0 * (phase - 0.52)))
        retreat = (0.18 if label == 3 else 0.55) * contact
        roots[0, :, 1] = -0.5 * control.separation
        roots[1, :, 1] = 0.5 * control.separation + retreat
        roots[:, :, 0] = 0.10 * control.speed * (phase - 0.5)
    return roots


def generate_one(reference: np.ndarray, label: int, rng: np.random.Generator,
                 min_frames: int, max_frames: int, noise_std: float,
                 yaw_limit: float) -> tuple[np.ndarray, MotionControl]:
    frames = int(rng.integers(min_frames, max_frames + 1))
    control = MotionControl(
        frames=frames,
        yaw=float(rng.uniform(-yaw_limit, yaw_limit)),
        scale=float(rng.uniform(0.88, 1.12)),
        speed=float(rng.uniform(0.50, 0.90) if label == 0 else rng.uniform(1.05, 1.80)),
        separation=float(rng.uniform(0.45, 0.95)),
        noise_std=noise_std,
    )
    local = local_templates(reference, frames) * control.scale
    keypoint = local + root_trajectories(label, control)[:, :, None, :]
    if label in INTERACTION_IDS:
        # Push/chase context needs two active people.  Light phase-dependent
        # upper-body jitter prevents identical copied templates.
        keypoint[1, :, 5:12] += rng.normal(0.0, noise_std * 2, keypoint[1, :, 5:12].shape)
    keypoint += rng.normal(0.0, noise_std, keypoint.shape).astype(np.float32)
    return rotate_z(keypoint, control.yaw).astype(np.float32), control


def passes_gate(keypoint: np.ndarray, label: int) -> tuple[bool, str]:
    if keypoint.shape[0] != 2 or keypoint.shape[2:] != (25, 3) or keypoint.shape[1] < 30:
        return False, "shape"
    if not np.isfinite(keypoint).all():
        return False, "non_finite"
    roots = keypoint[:, :, 0, :]
    velocity = np.linalg.norm(np.diff(roots, axis=1), axis=-1)
    if float(velocity.max()) > 0.25:
        return False, "root_velocity"
    primary_displacement = float(np.linalg.norm(roots[0, -1] - roots[0, 0]))
    if label in (0, 1) and primary_displacement < 0.35:
        return False, "insufficient_travel"
    if label in INTERACTION_IDS:
        distance = np.linalg.norm(roots[0] - roots[1], axis=-1)
        if float(distance.min()) < 0.12 or float(distance.max()) > 2.5:
            return False, "interaction_distance"
        if label in (3, 5):
            before = float(distance[: len(distance) // 3].mean())
            after = float(distance[-len(distance) // 3:].mean())
            minimum = 0.04 if label == 3 else 0.16
            if after - before < minimum:
                return False, "missing_push_retreat"
    return True, "ok"


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if set(payload.get("split", {})) != {"train", "val", "test"}:
        raise ValueError("Annotation must contain exactly train/val/test splits")
    return payload


def main() -> None:
    args = parse_args()
    counts = parse_counts(args.counts)
    if args.min_frames < 30 or args.max_frames < args.min_frames:
        raise ValueError("Require 30 <= --min-frames <= --max-frames")
    if args.noise_std < 0 or args.yaw_radians < 0 or args.max_attempts < 1:
        raise ValueError("Noise, yaw, and max attempts must be non-negative/positive")
    payload = load_annotation(args.input_annotation)
    annotations = list(payload["annotations"])
    by_id = {row["frame_dir"]: row for row in annotations}
    train_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in payload["split"]["train"]:
        row = by_id[sample_id]
        label = int(row["label"])
        if label not in range(len(LABELS)):
            raise ValueError(f"Unexpected label {label} in {sample_id}")
        train_by_label[label].append(row)
    rng = np.random.default_rng(args.seed)
    generated, manifest_rows, rejected = [], [], Counter()
    for label, target in counts.items():
        references = train_by_label[label]
        if target and not references:
            raise RuntimeError(f"No train references for {LABELS[label]}")
        accepted = 0
        attempts = 0
        while accepted < target:
            attempts += 1
            if attempts > target * args.max_attempts:
                raise RuntimeError(f"Quality gate stalled for {LABELS[label]}: {accepted}/{target}")
            reference = references[int(rng.integers(len(references)))]
            keypoint, control = generate_one(
                np.asarray(reference["keypoint"], dtype=np.float32), label, rng,
                args.min_frames, args.max_frames, args.noise_std, args.yaw_radians,
            )
            accepted_gate, reason = passes_gate(keypoint, label)
            if not accepted_gate:
                rejected[f"{LABELS[label]}:{reason}"] += 1
                continue
            frame_dir = f"synthetic_v1/{LABELS[label]}/{accepted:05d}_seed{args.seed}"
            generated.append({
                "frame_dir": frame_dir,
                "total_frames": int(keypoint.shape[1]),
                "label": label,
                "keypoint": keypoint,
            })
            manifest_rows.append({
                "frame_dir": frame_dir,
                "label": LABELS[label],
                "source_train_id": reference["frame_dir"],
                "control": control.__dict__,
                "quality_gate": reason,
            })
            accepted += 1
        print(f"generated {LABELS[label]}: {accepted}", flush=True)
    output = {
        "split": {
            "train": list(payload["split"]["train"]) + [row["frame_dir"] for row in generated],
            "val": list(payload["split"]["val"]),
            "test": list(payload["split"]["test"]),
        },
        "annotations": annotations + generated,
    }
    args.output_annotation.parent.mkdir(parents=True, exist_ok=True)
    with args.output_annotation.open("wb") as stream:
        pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "campus6_synthetic_ntu25.v1",
        "seed": args.seed,
        "input_annotation": str(args.input_annotation),
        "output_annotation": str(args.output_annotation),
        "synthetic_counts": dict(sorted(Counter(LABELS[row["label"]] for row in generated).items())),
        "train_before": len(payload["split"]["train"]),
        "train_after": len(output["split"]["train"]),
        "val_unchanged": payload["split"]["val"] == output["split"]["val"],
        "test_unchanged": payload["split"]["test"] == output["split"]["test"],
        "rejected": dict(sorted(rejected.items())),
        "controls": {
            "frames": [args.min_frames, args.max_frames], "noise_std": args.noise_std,
            "yaw_radians": args.yaw_radians, "counts": args.counts,
        },
        "limitation": "Synthetic samples are train-only augmentation, not evidence of real-world generalization.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
