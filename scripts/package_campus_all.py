"""Package the original 358-sample Campus6 features and RGB videos.

The source annotation file stores the original COCO-17 arrays inline.  This
utility exports one compressed ``.npz`` feature per annotation and copies the
digest-verified RGB video into a matching ``campus_all/video`` entry.  The
manifest is authoritative, so consumers do not need to infer paths from
filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)
FRAME_DIR_DIGEST = re.compile(r"^(.*)__([0-9a-f]{12})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_annotations(path: Path) -> tuple[dict, list[dict], dict[str, str]]:
    payload = pickle.loads(path.read_bytes())
    annotations = list(payload.get("annotations") or [])
    if len(annotations) != 358:
        raise ValueError(f"expected 358 annotations, got {len(annotations)}")
    split_for: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for frame_dir in (payload.get("split") or {}).get(split, []):
            split_for[str(frame_dir)] = split
    return payload, annotations, split_for


def resolve_videos(annotations: list[dict], root: Path) -> list[Path]:
    index = {path.stem: path for path in root.rglob("*.mp4")}
    resolved: list[Path] = []
    for annotation in annotations:
        frame_dir = str(annotation["frame_dir"])
        name = Path(frame_dir).name
        match = FRAME_DIR_DIGEST.match(name)
        if not match:
            raise ValueError(f"frame_dir has no 12-hex digest suffix: {frame_dir}")
        stem, digest = match.groups()
        video = index.get(stem)
        if video is None:
            raise FileNotFoundError(f"RGB video not found for {frame_dir}")
        actual = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
        if actual != digest:
            raise ValueError(
                f"video digest mismatch for {frame_dir}: expected {digest}, got {actual}"
            )
        resolved.append(video)
    return resolved


def sample_id(index: int, frame_dir: str) -> str:
    digest = hashlib.sha256(frame_dir.encode("utf-8")).hexdigest()[:8]
    return f"campus6_{index:04d}_{digest}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    payload, annotations, split_for = load_annotations(args.annotations)
    videos = resolve_videos(annotations, args.video_root)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty or absent: {output}")
    (output / "feature").mkdir(parents=True, exist_ok=True)
    (output / "video").mkdir(parents=True, exist_ok=True)

    annotation_copy = output / "annotations_with_all.pkl"
    shutil.copy2(args.annotations, annotation_copy)
    rows: list[dict] = []
    for index, (annotation, video) in enumerate(zip(annotations, videos)):
        frame_dir = str(annotation["frame_dir"])
        label_id = int(annotation["label"])
        identifier = sample_id(index, frame_dir)
        feature_path = output / "feature" / f"{identifier}.npz"
        video_path = output / "video" / f"{identifier}.mp4"
        keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
        keypoint_score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
        np.savez_compressed(
            feature_path,
            schema_version=np.asarray("rtmpose_coco17_2d.v1"),
            frame_dir=np.asarray(frame_dir),
            label=np.asarray(label_id, dtype=np.int64),
            keypoint=keypoint,
            keypoint_score=keypoint_score,
            valid_mask=keypoint_score >= 0.20,
            total_frames=np.asarray(int(annotation["total_frames"]), dtype=np.int32),
            img_shape=np.asarray(annotation.get("img_shape", (1, 1)), dtype=np.int32),
            extractor_id=np.asarray("original_annotations_with_all.v1"),
        )
        shutil.copy2(video, video_path)
        rows.append({
            "id": identifier,
            "annotation_index": index,
            "frame_dir": frame_dir,
            "label_id": label_id,
            "label": LABELS[label_id],
            "split": split_for.get(frame_dir, "all"),
            "feature": str(feature_path.relative_to(output)),
            "video": str(video_path.relative_to(output)),
            "feature_sha256": sha256(feature_path),
            "video_sha256": sha256(video_path),
            "feature_bytes": feature_path.stat().st_size,
            "video_bytes": video_path.stat().st_size,
        })

    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    checksums = output / "SHA256SUMS"
    checksum_files = [annotation_copy, manifest, *sorted((output / "feature").glob("*.npz")), *sorted((output / "video").glob("*.mp4"))]
    checksums.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in checksum_files),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "campus6_all_release.v1",
        "sample_count": len(rows),
        "feature_count": len(list((output / "feature").glob("*.npz"))),
        "video_count": len(list((output / "video").glob("*.mp4"))),
        "feature_source": str(args.annotations),
        "video_source": str(args.video_root),
        "labels": list(LABELS),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "note": "Features are the original arrays from annotations_with_all.pkl; no re-extraction was performed.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Campus6 all-sample release\n\n"
        "This package contains the original 358 Campus6 COCO-17 features and "
        "their digest-verified RGB videos. `manifest.jsonl` is the authoritative "
        "one-to-one mapping between `feature/` and `video/`. Verify all files "
        "with `sha256sum -c SHA256SUMS`.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
