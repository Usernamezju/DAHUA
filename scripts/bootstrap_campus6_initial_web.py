#!/usr/bin/env python3
"""Materialize the official Campus6 skeleton baseline for the Web service.

The official training annotation stores the COCO-17 sequence directly.  This
tool deliberately creates only skeleton features and rendered skeleton videos:
it never copies or exposes source RGB video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.render_rtmpose17_pose import build_parser, render


LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", required=True, help="official annotations_with_all.pkl")
    value.add_argument("--runtime-root", required=True)
    value.add_argument("--render-pose", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def sample_id(frame_dir: str) -> str:
    digest = hashlib.sha256(frame_dir.encode("utf-8")).hexdigest()[:16]
    return "campus6_" + digest


def split_index(value: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, names in (value.get("split") or {}).items():
        for name in names:
            result[str(name)] = str(split)
    return result


def write_feature(path: Path, annotation: dict, *, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        return
    keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
    score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
    if keypoint.ndim != 4 or keypoint.shape[0] not in (1, 2) or keypoint.shape[2:] != (17, 2):
        raise ValueError("official annotation must contain [M,T,17,2] COCO-17 keypoints")
    if score.shape != keypoint.shape[:3]:
        raise ValueError("keypoint_score shape is incompatible with keypoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("rtmpose_coco17_2d.v1"),
        keypoint=keypoint,
        keypoint_score=score,
        valid_mask=score >= 0.20,
        fps=np.asarray(10.0, dtype=np.float32),
        source_fps=np.asarray(10.0, dtype=np.float32),
        total_frames=np.asarray(keypoint.shape[1], dtype=np.int32),
        width=np.asarray(1, dtype=np.int32),
        height=np.asarray(1, dtype=np.int32),
    )


def render_pose(feature: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.is_file() and not overwrite:
        return
    args = build_parser().parse_args([
        "--feature", str(feature), "--output", str(destination),
    ])
    render(args)


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    source = Path(args.annotations).expanduser().resolve()
    root = Path(args.runtime_root).expanduser().resolve()
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    annotations = list(payload.get("annotations") or [])
    if not annotations:
        raise ValueError("annotations_with_all.pkl contains no annotations")
    splits = split_index(payload)
    artifacts = root / "artifacts"
    rows = []
    for annotation in annotations:
        frame_dir = str(annotation["frame_dir"])
        label_index = int(annotation["label"])
        if not 0 <= label_index < len(LABELS):
            raise ValueError(f"invalid Campus6 label index: {label_index}")
        identifier = sample_id(frame_dir)
        feature = artifacts / "features" / f"{identifier}.npz"
        pose = artifacts / "pose_videos" / f"{identifier}.mp4"
        write_feature(feature, annotation, overwrite=args.overwrite)
        if args.render_pose:
            render_pose(feature, pose, overwrite=args.overwrite)
        rows.append({
            "clip_id": identifier,
            # The manifest path is only an internal skeleton artifact anchor;
            # the API never returns it and never stores source RGB paths.
            "path": str(pose),
            "source_dataset": "Campus6_initial",
            "source_label": LABELS[label_index],
            "suggested_coarse_label": "",
            "manual_label": "",
            "keep": "",
            "notes": "official_initial_split=" + splits.get(frame_dir, "unknown"),
        })
    manifest = root / "campus6_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} Campus6 skeleton records to {manifest}")


if __name__ == "__main__":
    main()
