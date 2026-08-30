#!/usr/bin/env python3
"""Build one CSV for manually relabeling downloaded behavior candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "clip_id",
    "path",
    "source_dataset",
    "source_label",
    "suggested_coarse_label",
    "manual_label",
    "keep",
    "notes",
)

KTH_SUGGESTIONS = {
    "walking": "normal_walk",
    "jogging": "normal_run",
    "running": "normal_run",
    "boxing": "conflict_contact_candidate",
}

LIMU_SUGGESTIONS = {
    "push": "push_unspecified",
    "punch": "conflict_contact_candidate",
    "pull": "conflict_contact_candidate",
    "kick": "conflict_contact_candidate",
    "hug": "light_contact_or_out_of_scope",
    "touch": "light_contact_or_out_of_scope",
    "handshake": "light_contact_or_out_of_scope",
    "handover": "light_contact_or_out_of_scope",
    "handclap": "light_contact_or_out_of_scope",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    return parser


def row(clip_id, path, dataset, source_label, suggestion):
    return {
        "clip_id": clip_id,
        "path": str(path.resolve()),
        "source_dataset": dataset,
        "source_label": source_label,
        "suggested_coarse_label": suggestion,
        "manual_label": "",
        "keep": "",
        "notes": "",
    }


def collect_behave(root):
    rows = []
    clips = root / "behave" / "prepared" / "clips"
    for path in sorted(clips.glob("*/*.mp4")):
        source_label = path.parent.name
        if source_label == "WalkTogether":
            suggestion = "normal_walk"
        elif source_label == "RunTogether":
            suggestion = "normal_run"
        elif source_label == "Chase":
            suggestion = "chase_unspecified"
        else:
            suggestion = "conflict_contact_candidate"
        rows.append(row(path.stem, path, "BEHAVE", source_label, suggestion))
    return rows


def collect_kth(root):
    rows = []
    extracted = root / "kth" / "extracted"
    for source_label, suggestion in KTH_SUGGESTIONS.items():
        for path in sorted((extracted / source_label).rglob("*.avi")):
            rows.append(row(f"kth_{path.stem}", path, "KTH", source_label, suggestion))
    return rows


def collect_limu(root):
    rows = []
    extracted = root / "limu" / "extracted" / "video-interaction"
    for path in sorted(extracted.glob("*.avi")):
        source_label = path.stem.rsplit("_", 1)[0]
        suggestion = LIMU_SUGGESTIONS.get(source_label, "out_of_scope")
        rows.append(row(f"limu_{path.stem}", path, "LIMU", source_label, suggestion))
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "manual_label_manifest.csv"
    rows = collect_behave(root) + collect_kth(root) + collect_limu(root)
    if not rows:
        raise RuntimeError(f"no candidate videos found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} candidates to {output}")


if __name__ == "__main__":
    main()
