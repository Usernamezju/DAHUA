"""Versioned prompt construction from structured, measurable evidence."""

from __future__ import annotations

import json
from typing import Any, Mapping

from dahua_cup.semantic_teacher.schemas import LABELS


PROMPT_VERSION = "campus6-v1.0"


def build_teacher_prompt(
    sample_id: str,
    semantic_graph: Mapping[str, Any],
    student_distribution: Mapping[str, float] | None = None,
    allowed_labels: tuple[str, ...] = LABELS,
    task: str = "campus6",
) -> str:
    if not sample_id:
        raise ValueError("sample_id is required")
    labels = tuple(str(label) for label in allowed_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("allowed_labels must be non-empty and unique")
    student = {
        label: float((student_distribution or {}).get(label, 0.0))
        for label in labels
        if label in (student_distribution or {})
    }
    payload = {
        "sample_id": sample_id,
        "task": task,
        "semantic_graph": semantic_graph,
        "student_distribution": student,
    }
    return f"""You are a weak-supervision teacher for closed-set video behavior analysis.
Prompt version: {PROMPT_VERSION}
Task: {task}
Allowed labels ({len(labels)}): {json.dumps(labels, ensure_ascii=False)}

Rules:
- Use only facts present in INPUT. Do not invent intent, impact force, fear, attacks, defense, or contact.
- Mark unavailable evidence as unknown. Student probabilities are a fallible hint and may be corrected.
- Every evidence item must cite an existing segment_id.
- Return one JSON object only, matching teacher_output.v1.
- label must be selected from Allowed labels.
- distribution may contain only the most likely labels, but returned probabilities must sum to 1; omitted allowed labels are treated as zero.
- label must be the argmax of distribution.
- confidence must equal distribution[label] after the listed probabilities sum to 1.
- Set needs_review=true for weak visibility, conflicting cues, or close alternatives.

INPUT:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

OUTPUT KEYS:
schema_version, sample_id, label, distribution, confidence, evidence,
counter_evidence, reason, needs_review
"""
