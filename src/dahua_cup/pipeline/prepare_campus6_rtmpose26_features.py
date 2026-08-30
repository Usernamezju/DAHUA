#!/usr/bin/env python3
"""Prepare the current reviewed Campus6 pool for RTMPose/ProtoGCN.

Unsplit candidates in the re-audit pool are byte copies of previously posed
videos but have new filenames.  This script links their existing feature file
under the new deterministic name.  A split child, in contrast, is always put
in ``missing`` and must be posed from its own clip; reusing a parent's pose
would leak frames outside the child segment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "normal_walking": "normal_walk", "normal_running": "normal_run",
    "playful_chasing": "playful_chase", "playful_pushing": "playful_push",
    "aggressive_chasing": "conflict_chase", "aggressive_pushing": "conflict_push",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def feature_path(features: Path, label: str, video: Path) -> Path:
    return features / label / f"{video.stem}__{digest(video)}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--missing-manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(args.pool / "candidate_manifests/probe_extraction_manifest_v3.csv")
    labels = {row["video_id"]: row["manual_label"].strip() for row in read_csv(args.pool / "candidate_manifests/manual_labels_v3.csv")}
    by_id = {row["video_id"]: row for row in manifest_rows}
    provenance = read_csv(args.pool / "candidate_manifests/hidden_provenance.csv")
    original_by_id = {row["new_video_id"]: Path(row["original_video_path"]) for row in provenance}
    existing_by_digest: dict[str, Path] = {}
    for path in args.features.rglob("*.npz"):
        # Existing extractor output is '<video_stem>__<sha1(path)[:12]>.npz'.
        suffix = path.stem.rsplit("__", 1)[-1]
        if len(suffix) == 12:
            existing_by_digest.setdefault(suffix, path)

    active = []
    for row in manifest_rows:
        label = LABEL_MAP.get(labels.get(row["video_id"], ""))
        video = Path(row.get("local_path", ""))
        if row.get("split_status") == "split_parent" or not label or not video.is_file():
            continue
        root_id, cursor, is_child = row["video_id"], row, bool(row.get("parent_video_id"))
        seen = {root_id}
        while cursor.get("parent_video_id"):
            parent_id = cursor["parent_video_id"]
            if parent_id in seen or parent_id not in by_id:
                break
            seen.add(parent_id)
            root_id, cursor = parent_id, by_id[parent_id]
        active.append({
            "video_id": row["video_id"], "video": str(video), "label": label,
            "parent_video_id": root_id, "source": row.get("source", "Campus6_ProtoGCN"),
            "is_child": is_child,
        })

    missing, linked = [], 0
    for item in active:
        video, target = Path(item["video"]), feature_path(args.features, item["label"], Path(item["video"]))
        if target.is_file():
            continue
        # Only unchanged, unsplit pool copies can safely reuse a source pose.
        source = original_by_id.get(item["video_id"])
        source_feature = existing_by_digest.get(digest(source)) if source and not item["is_child"] else None
        if source_feature and source_feature.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source_feature)
            linked += 1
        else:
            missing.append(item)

    def dump(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({k: v for k, v in row.items() if k != "is_child"}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    dump(args.output_manifest, active)
    dump(args.missing_manifest, missing)
    print(json.dumps({
        "active": len(active), "linked_existing_features": linked, "already_present": len(active) - linked - len(missing),
        "needs_pose_extraction": len(missing), "missing_by_class": dict(Counter(row["label"] for row in missing)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
