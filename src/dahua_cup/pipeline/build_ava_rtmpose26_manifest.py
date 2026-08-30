"""Build the labelled six-class manifest for AVA/Kinetics RTMPose extraction.

The manual review table is authoritative.  ``unusable`` is deliberately
excluded rather than being treated as a seventh action class.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABEL_MAP = {
    "normal_walking": "normal_walk",
    "normal_running": "normal_run",
    "playful_chasing": "playful_chase",
    "playful_pushing": "playful_push",
    "aggressive_chasing": "conflict_chase",
    "aggressive_pushing": "conflict_push",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.labels_csv.open(newline="", encoding="utf-8") as handle:
        labels = {
            row["video_id"]: row["manual_label"].strip()
            for row in csv.DictReader(handle)
        }

    rows = []
    with args.extraction_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # A split parent is retained for provenance but must never enter
            # extraction or training together with its derived child clips.
            if row.get("split_status") == "split_parent":
                continue
            label = LABEL_MAP.get(labels.get(row["video_id"], ""))
            video = Path(row.get("local_path", ""))
            if not label or not video.is_file():
                continue
            rows.append({
                "video_id": row["video_id"],
                "video": str(video),
                "label": label,
                "parent_video_id": row.get("parent_video_id") or row["video_id"],
                "source": row.get("source", "AVA-Kinetics"),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {name: sum(row["label"] == name for row in rows) for name in LABEL_MAP.values()}
    print(json.dumps({"samples": len(rows), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
