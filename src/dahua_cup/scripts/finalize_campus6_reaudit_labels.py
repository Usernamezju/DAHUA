#!/usr/bin/env python3
"""Preserve manual review labels and backfill old labels for the re-audit pool."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


POOL_DEFAULT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/campus6_protogcn_reaudit_v1")
LABEL_FIELDS = [
    "video_id", "candidate_group", "manual_label", "interaction_direction", "trajectory", "contact",
    "after_contact", "aggression_evidence", "playful_evidence", "scene", "pose_quality",
    "annotator_note", "vlm_decision", "updated_at",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=POOL_DEFAULT)
    parser.add_argument("--snapshot-name", default="completed_reaudit_snapshot_v1")
    args = parser.parse_args()
    pool = args.pool.resolve()
    manifests = pool / "candidate_manifests"
    manifest_path, labels_path = manifests / "probe_extraction_manifest_v3.csv", manifests / "manual_labels_v3.csv"
    manifest = read_csv(manifest_path)
    labels = read_csv(labels_path)
    by_key = {(row.get("video_id", ""), row.get("candidate_group", "")): dict(row) for row in labels}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preserved, autofilled, still_unlabeled = 0, 0, 0
    for item in manifest:
        key = (item["video_id"], item["candidate_group"])
        existing = by_key.get(key)
        if existing and existing.get("manual_label", "") not in ("", "unlabeled"):
            preserved += 1
            continue
        original = item.get("kinetics_label", "").strip()
        # Only imported ProtoGCN candidates have an earlier human label.
        # Desktop/manual uploads deliberately remain awaiting review.
        if item.get("source") == "Campus6_ProtoGCN" and original and original != "unlabeled":
            row = {field: "" for field in LABEL_FIELDS}
            if existing:
                row.update(existing)
            row.update({
                "video_id": key[0], "candidate_group": key[1], "manual_label": original,
                "annotator_note": (row.get("annotator_note", "") + " | " if row.get("annotator_note", "") else "")
                + "Backfilled from original manual label.",
                "updated_at": now,
            })
            by_key[key] = row
            autofilled += 1
        else:
            still_unlabeled += 1
    output_rows = [by_key[key] for key in sorted(by_key)]
    backup = labels_path.with_name("manual_labels_v3.before_backfill.csv")
    shutil.copy2(labels_path, backup)
    write_csv(labels_path, output_rows, LABEL_FIELDS)

    snapshot = pool / args.snapshot_name
    if snapshot.exists():
        raise SystemExit(f"snapshot already exists: {snapshot}")
    snapshot.mkdir(parents=True)
    for name in ("manual_labels_v3.csv", "probe_extraction_manifest_v3.csv", "vlm_soft_review_v3.csv", "review_order_v3.csv"):
        source = manifests / name
        if source.exists():
            shutil.copy2(source, snapshot / name)
    summary = {
        "created_at": now, "pool": str(pool), "manual_labels_preserved": preserved,
        "old_labels_backfilled": autofilled, "manual_uploads_still_unlabeled": still_unlabeled,
        "snapshot": str(snapshot),
    }
    (snapshot / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
