"""Atomic persistence and review write-back for versioned pseudo-label JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from dahua_cup.semantic_teacher.schemas import LABELS


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"pseudo-label dataset not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_back_review(
    path: str | Path,
    sample_id: str,
    review: Mapping,
) -> dict:
    """Write a human decision into its pseudo-label record atomically."""
    source = Path(path)
    rows = _read(source)
    matched = None
    for row in rows:
        if str(row.get("sample_id")) != sample_id:
            continue
        previous = {
            "label": row.get("label"),
            "soft_label": row.get("soft_label"),
            "status": row.get("status"),
            "source": row.get("source"),
            "teacher_soft_label": row.get("teacher_soft_label"),
            "label_source": row.get("label_source"),
        }
        final_label = str(review["final_label"])
        value = dict(review)
        value["previous"] = previous
        history = list(row.get("review_history") or [])
        if row.get("review"):
            history.append(row["review"])
        row["review_history"] = history
        row["review"] = value
        row["label_source"] = "human_review"
        if final_label in LABELS:
            if not row.get("teacher_soft_label"):
                row["teacher_soft_label"] = list(row.get("soft_label") or [])
            row["label"] = final_label
            row["soft_label"] = [
                float(label == final_label) for label in LABELS
            ]
            row["status"] = "accepted"
            row["source"] = "human_review"
        else:
            row["status"] = "rejected"
        matched = row
        break
    if matched is None:
        raise KeyError(sample_id)
    _write_atomic(source, rows)
    return matched


def revert_review(path: str | Path, sample_id: str) -> dict:
    """Restore the pre-review pseudo-label state after an audit undo."""
    source = Path(path)
    rows = _read(source)
    matched = None
    for row in rows:
        if str(row.get("sample_id")) != sample_id:
            continue
        review = row.get("review") or {}
        previous = review.get("previous")
        if not previous:
            raise ValueError(f"pseudo-label has no review to revert: {sample_id}")
        for key, value in previous.items():
            if value is None and key in {"teacher_soft_label", "label_source"}:
                row.pop(key, None)
            else:
                row[key] = value
        history = list(row.get("review_history") or [])
        row["review"] = history.pop() if history else None
        row["review_history"] = history
        matched = row
        break
    if matched is None:
        raise KeyError(sample_id)
    _write_atomic(source, rows)
    return matched
