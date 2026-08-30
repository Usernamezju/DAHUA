#!/usr/bin/env python3
"""Render compact contact sheets for an explicit Campus6 video review list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dahua_cup.dataset_construction.contact_sheets import create_contact_sheets_isolated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True,
                        help="JSON list of {id, label, video} review items")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = {
        "total_frames": 8, "uniform_frames": 6, "motion_frames": 2,
        "scan_frames": 24, "frames_per_sheet": 8,
        "cell_width": 240, "cell_height": 180, "jpeg_quality": 88,
        "worker_timeout_seconds": 120,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for item in json.loads(args.items.read_text(encoding="utf-8")):
        identifier = item["id"]
        destination = args.output / identifier
        try:
            sheets, metadata = create_contact_sheets_isolated(
                Path(item["video"]), destination, config
            )
            report.append({**item, "status": "ok", "sheets": [str(path) for path in sheets],
                           "timestamps_seconds": metadata["timestamps_seconds"]})
        except Exception as error:
            report.append({**item, "status": "error", "error": str(error)})
    (args.output / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
