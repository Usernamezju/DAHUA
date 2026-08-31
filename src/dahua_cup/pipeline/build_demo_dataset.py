"""Build the standalone demo dataset under ``dataset/`` from server artefacts.

Copies the full Campus6 358-sample baseline (skeleton annotations, M1KD
predictions and the digest-verified RGB source videos) plus a deterministic
25-sample Campus6 selection and a 25-sample KTH selection into one dataset
directory, and writes ``selection.json`` for the demo web backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)

FRAME_DIR_DIGEST = re.compile(r"^(.*)__([0-9a-f]{12})$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="annotations_with_all.pkl")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="M1KD.eval_all.pkl (optional sidecar: .json)")
    parser.add_argument("--video-root", type=Path, required=True,
                        help="campus6_protogcn_reaudit_v1/videos_probe directory")
    parser.add_argument("--kth-root", type=Path, required=True,
                        help="KTH extracted directory with boxing/jogging/running/walking")
    parser.add_argument("--output", type=Path, required=True,
                        help="dataset directory to create")
    parser.add_argument("--campus6-count", type=int, default=25)
    parser.add_argument("--kth-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def load_annotations(path: Path):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    annotations = list(payload.get("annotations") or [])
    split_for = {}
    for split in ("train", "val", "test"):
        for name in payload.get("split", {}).get(split, []):
            split_for[str(name)] = split
    return annotations, split_for


def resolve_rgb_video(frame_dir: str, video_index: dict[str, Path]) -> Path:
    """Recover the RGB video for one annotation via the sha1 path digest."""
    name = frame_dir.split("/")[-1]
    match = FRAME_DIR_DIGEST.match(name)
    if not match:
        raise ValueError(f"frame_dir has no 12-hex digest suffix: {frame_dir}")
    stem, digest = match.group(1), match.group(2)
    video = video_index.get(stem)
    if video is None:
        raise FileNotFoundError(f"RGB video not found for {frame_dir}")
    actual = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
    if actual != digest:
        raise ValueError(f"digest mismatch for {frame_dir}: expected {digest}, got {actual}")
    return video


def distribute(count: int, class_names: list[str], class_totals: dict[str, int]):
    """Split `count` across classes with the largest-remainder method."""
    total = sum(class_totals.values())
    quotas = {name: count * class_totals[name] // total for name in class_names}
    remainders = sorted(
        class_names,
        key=lambda n: (count * class_totals[n] % total, class_totals[n]),
        reverse=True,
    )
    for name in remainders[: count - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def select_campus6(annotations, split_for, count: int, seed: int):
    """Class-balanced selection preferring the official test split."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for index, annotation in enumerate(annotations):
        by_class[LABELS[int(annotation["label"])]].append(index)
    class_totals = {name: len(by_class[name]) for name in LABELS}
    quotas = distribute(count, LABELS, class_totals)
    selected = []
    for name in LABELS:
        candidates = list(by_class[name])
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: {"test": 0, "val": 1, "train": 2}[
            split_for.get(str(annotations[i]["frame_dir"]), "train")])
        selected.extend(candidates[:quotas[name]])
    return sorted(selected)


def select_kth(kth_root: Path, count: int, seed: int) -> list[dict]:
    """Select `count` KTH clips, balanced across the four classes."""
    rng = random.Random(seed)
    classes = ["boxing", "jogging", "running", "walking"]
    quotas = distribute(count, classes, {name: 100 for name in classes})
    rows = []
    for name in classes:
        files = sorted((kth_root / name).glob("*.avi"))
        # Spread over persons and rotate scenarios, then shuffle deterministically.
        rng.shuffle(files)
        person_seen = set()
        spread = []
        for path in files:
            person = path.stem.split("_")[0][-2:]
            if person in person_seen:
                continue
            person_seen.add(person)
            spread.append(path)
            if len(spread) >= quotas[name]:
                break
        if len(spread) < quotas[name]:
            spread.extend(files[:quotas[name] - len(spread)])
        rows.extend({
            "path": path,
            "label": name,
            "person": path.stem.split("_")[0][-2:],
            "scenario": path.stem.split("_")[-2],
        } for path in spread)
    rows.sort(key=lambda row: (classes.index(row["label"]), row["person"]))
    return rows


def main():
    args = parse_args()
    annotations, split_for = load_annotations(args.annotations)
    if len(annotations) != 358:
        print(f"warning: annotations contain {len(annotations)} samples, expected 358",
              file=sys.stderr)
    video_index = {video.stem: video for video in args.video_root.rglob("*.mp4")}
    output = args.output
    baseline_dir = output / "campus6_baseline"
    videos_dir = output / "videos"
    campus6_dir = videos_dir / "campus6"
    kth_dir = videos_dir / "kth"
    for directory in (baseline_dir, campus6_dir, kth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Baseline artefacts (annotations, predictions, metrics sidecars).
    for name in ("annotations_with_all.pkl", "annotations_with_all.summary.json",
                 "M1KD.eval_all.pkl", "M1KD.eval_all.pkl.json", "M1KD.metrics.json",
                 "comparison.json", "experiment_contract.json"):
        source = args.annotations.parent / name
        if source.is_file():
            shutil.copy2(source, baseline_dir / name)

    # 2. Full 358-sample RGB videos, flattened and digest-verified.
    video_map = {}
    for annotation in annotations:
        frame_dir = str(annotation["frame_dir"])
        video = resolve_rgb_video(frame_dir, video_index)
        label = LABELS[int(annotation["label"])]
        destination = campus6_dir / f"{label}__{video.stem}{video.suffix}"
        if not destination.is_file():
            shutil.copy2(video, destination)
        video_map[frame_dir] = str(destination.relative_to(output))

    # 3. Deterministic 25-sample Campus6 selection.
    campus6_ids = select_campus6(annotations, split_for, args.campus6_count, args.seed)
    campus6_rows = []
    for index in campus6_ids:
        annotation = annotations[index]
        frame_dir = str(annotation["frame_dir"])
        campus6_rows.append({
            "id": f"campus6_{index:04d}_{hashlib.sha256(frame_dir.encode()).hexdigest()[:8]}",
            "label": LABELS[int(annotation["label"])],
            "split": split_for.get(frame_dir, "train"),
            "annotation_index": index,
            "rgb": video_map[frame_dir],
            "source": "campus6",
        })

    # 4. Deterministic 25-sample KTH selection and copy.
    kth_rows = []
    for row in select_kth(args.kth_root, args.kth_count, args.seed):
        destination = kth_dir / row["path"].name
        if not destination.is_file():
            shutil.copy2(row["path"], destination)
        kth_rows.append({
            "id": f"kth_{row['path'].stem}",
            "label": row["label"],
            "person": row["person"],
            "scenario": row["scenario"],
            "rgb": str(destination.relative_to(output)),
            "source": "kth",
        })

    # 5. selection.json summary.
    selection = {
        "schema_version": "demo_selection.v1",
        "seed": args.seed,
        "created_from": {
            "annotations": str(args.annotations),
            "predictions": str(args.predictions) if args.predictions else None,
            "video_root": str(args.video_root),
            "kth_root": str(args.kth_root),
        },
        "campus6": campus6_rows,
        "kth": kth_rows,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "campus6_total": len(annotations),
        "campus6_videos_copied": len(video_map),
        "campus6_selected": len(campus6_rows),
        "campus6_selected_by_class": dict(Counter(r["label"] for r in campus6_rows)),
        "campus6_selected_by_split": dict(Counter(r["split"] for r in campus6_rows)),
        "kth_selected": len(kth_rows),
        "kth_selected_by_class": dict(Counter(r["label"] for r in kth_rows)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (output / "selection.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
