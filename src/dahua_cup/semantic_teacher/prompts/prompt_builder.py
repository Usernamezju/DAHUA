"""Versioned prompt construction from structured, measurable evidence."""

from __future__ import annotations

import json
from typing import Any, Mapping

from dahua_cup.semantic_teacher.schemas import LABELS


PROMPT_VERSION = "campus6-v1.3"


def build_teacher_prompt(
    sample_id: str,
    semantic_graph: Mapping[str, Any],
    allowed_labels: tuple[str, ...] = LABELS,
    task: str = "campus6",
) -> str:
    if not sample_id:
        raise ValueError("sample_id is required")
    labels = tuple(str(label) for label in allowed_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("allowed_labels must be non-empty and unique")
    payload = {
        "sample_id": sample_id,
        "task": task,
        "semantic_graph": semantic_graph,
    }
    return f"""You are a weak-supervision teacher for closed-set video behavior analysis.
Prompt version: {PROMPT_VERSION}
Task: {task}
Allowed labels ({len(labels)}): {json.dumps(labels, ensure_ascii=False)}

Rules:
- Use only facts present in INPUT. Do not invent intent, impact force, fear, attacks, defense, or contact.
- Judge independently from the skeleton video and measured INPUT only. You receive no student-model label or probability and must not assume one.
- If INPUT.semantic_graph.persons contains two or more people, this is an interaction clip: choose only one of playful_chase, playful_push, conflict_chase, or conflict_push. In that case normal_walk and normal_run must be omitted from distribution (equivalent to probability 0), even if one person moves slowly or appears stationary.
- evidence must contain at least one item, and every item must cite an existing segment_id.
- If no positive behavior cue is available, cite a measured INPUT fact (for example low motion, missing relation, or insufficient coverage) as the evidence and set needs_review=true.
- Return one JSON object only, matching teacher_output.v1.
- schema_version must be exactly "teacher_output.v1".
- Do not output label: the label is derived from distribution (its argmax) and never accepted from the model.
- distribution may contain only the most likely labels, but returned probabilities must sum to 1; omitted allowed labels are treated as zero.
- confidence must equal the largest probability in distribution after the listed probabilities sum to 1. These are your own relative assessments, not copied scores from another model.
- Set needs_review=true for weak visibility, conflicting cues, or close alternatives.

INPUT:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

OUTPUT KEYS:
schema_version, sample_id, distribution, confidence, evidence,
counter_evidence, reason, needs_review
"""
