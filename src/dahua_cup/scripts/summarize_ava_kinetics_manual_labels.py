#!/usr/bin/env python3
"""Summarize the read-only state of an AVA/Kinetics manual-review CSV.

The script never writes the input CSV.  It reports the label distribution and
per-candidate-group breakdown needed before another candidate batch is added.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_LABELS = (
    "normal_walking",
    "normal_running",
    "playful_chasing",
    "aggressive_chasing",
    "playful_pushing",
    "aggressive_pushing",
)
UNRESOLVED_LABELS = frozenset(
    {
        "",
        "unlabeled",
        "ambiguous",
        "uncertain",
        "unclear",
        "not_relevant",
        "irrelevant",
        "unusable",
        "bad_video",
    }
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _field_or_error(fieldnames: Iterable[str] | None, name: str) -> None:
    if not fieldnames or name not in fieldnames:
        available = ", ".join(fieldnames or ())
        raise ValueError(
            f"CSV must contain a {name!r} column; available columns: {available}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/"
            "candidate_manifests/manual_labels_v3.csv"
        ),
        help="Manual-label CSV to inspect (default: the AVA/Kinetics v3 file).",
    )
    parser.add_argument(
        "--label-column",
        default="manual_label",
        help="Name of the final-label column (default: manual_label).",
    )
    parser.add_argument(
        "--group-column",
        default="candidate_group",
        help="Name of the candidate-source group column (default: candidate_group).",
    )
    parser.add_argument(
        "--target-label",
        action="append",
        default=None,
        help=(
            "A label counted as a six-class usable sample. Repeat this option "
            "to override the defaults."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.labels.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manual-label CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _field_or_error(reader.fieldnames, args.label_column)
        _field_or_error(reader.fieldnames, args.group_column)
        rows = list(reader)

    targets = tuple(args.target_label or DEFAULT_LABELS)
    target_set = set(targets)
    label_counts = Counter(_clean(row.get(args.label_column)) for row in rows)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_counts[_clean(row.get(args.group_column)) or "<missing>"][_clean(
            row.get(args.label_column)
        )] += 1

    completed = sum(count for label, count in label_counts.items() if label not in {"", "unlabeled"})
    usable = sum(label_counts[label] for label in targets)
    unresolved = sum(
        count for label, count in label_counts.items() if label.lower() in UNRESOLVED_LABELS
    )

    print(f"Input: {path}")
    print(f"Total records: {len(rows)}")
    print(f"Completed reviews: {completed}")
    print(f"Six-class usable reviews: {usable}")
    print(f"Unresolved / excluded reviews: {unresolved}")
    print()
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        shown = label or "<empty>"
        marker = " [six-class]" if label in target_set else ""
        print(f"  {shown}: {count}{marker}")

    print()
    print("Candidate-group summary:")
    for group in sorted(group_counts):
        counts = group_counts[group]
        total = sum(counts.values())
        group_usable = sum(counts[label] for label in targets)
        print(f"  {group}: {total} total, {group_usable} six-class usable")
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {_clean(label) or '<empty>'}: {count}")

    unknown_targets = sorted(target_set - set(label_counts))
    if unknown_targets:
        print()
        print("Note: target labels not present in this CSV: " + ", ".join(unknown_targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
