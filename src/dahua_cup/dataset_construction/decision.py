"""Schema validation and conservative multi-model consensus for Campus6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CAMPUS6_LABELS: Tuple[str, ...] = (
    "normal_walk",
    "normal_run",
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
)
TARGET_PER_CLASS = 500

NON_FULL_LABELS = {
    "uncertain_intent",
    "crowded",
    "incomplete_action",
    "irrelevant",
    "bad_video",
    "no_action",
}
REJECT_LABELS = {
    "crowded",
    "incomplete_action",
    "irrelevant",
    "bad_video",
    "no_action",
}
MOTION_VALUES = {"walk", "run", "chase", "push", "other", "uncertain"}
INTENT_VALUES = {"neutral", "playful", "conflict", "unknown"}
GRADE_VALUES = {"A", "B", "C"}
DECISION_VALUES = {"accept_full", "accept_partial", "manual_review", "reject"}

EXPECTED_COMPONENTS = {
    "normal_walk": ("walk", "neutral"),
    "normal_run": ("run", "neutral"),
    "playful_chase": ("chase", "playful"),
    "playful_push": ("push", "playful"),
    "conflict_chase": ("chase", "conflict"),
    "conflict_push": ("push", "conflict"),
}

EVIDENCE_FIELDS = (
    "unilateral_pursuit",
    "role_exchange",
    "repeated_return",
    "escape_behavior",
    "defensive_pose",
    "physical_contact",
    "strong_displacement",
    "loss_of_balance",
)


@dataclass(frozen=True)
class ParsedReview:
    raw: Dict[str, Any]
    valid: bool
    errors: Tuple[str, ...]

    @property
    def label(self) -> str:
        return str(self.raw.get("campus6_label", ""))

    @property
    def motion(self) -> str:
        return str(self.raw.get("motion_primitive", ""))

    @property
    def grade(self) -> str:
        return str(self.raw.get("evidence_grade", ""))

    @property
    def decision(self) -> str:
        return str(self.raw.get("decision", ""))


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract one JSON object from a model response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def _person_count(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            return int(digits)
    return None


def validate_review(value: Dict[str, Any]) -> ParsedReview:
    errors: List[str] = []
    label = value.get("campus6_label")
    motion = value.get("motion_primitive")
    intent = value.get("intent")
    grade = value.get("evidence_grade")
    decision = value.get("decision")
    person_count = _person_count(value.get("primary_person_count"))

    if label not in set(CAMPUS6_LABELS) | NON_FULL_LABELS:
        errors.append("invalid campus6_label")
    if motion not in MOTION_VALUES:
        errors.append("invalid motion_primitive")
    if intent not in INTENT_VALUES:
        errors.append("invalid intent")
    if grade not in GRADE_VALUES:
        errors.append("invalid evidence_grade")
    if decision not in DECISION_VALUES:
        errors.append("invalid decision")
    if person_count is None or person_count < 0:
        errors.append("invalid primary_person_count")
    if not isinstance(value.get("third_person_present"), bool):
        errors.append("third_person_present must be boolean")
    if value.get("person_visibility") not in {"good", "acceptable", "poor"}:
        errors.append("invalid person_visibility")
    if value.get("action_completeness") not in {"complete", "partial", "missing"}:
        errors.append("invalid action_completeness")
    if not isinstance(value.get("reason"), str) or not value.get("reason", "").strip():
        errors.append("reason is required")

    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
    else:
        for field in EVIDENCE_FIELDS:
            if not isinstance(evidence.get(field), bool):
                errors.append(f"evidence.{field} must be boolean")

    if label in CAMPUS6_LABELS:
        expected_motion, expected_intent = EXPECTED_COMPONENTS[label]
        if motion != expected_motion or intent != expected_intent:
            errors.append("full label disagrees with motion or intent")
        if decision != "accept_full":
            errors.append("full label requires accept_full")
        if grade != "A":
            errors.append("full label requires evidence grade A")
        if value.get("action_completeness") != "complete":
            errors.append("full label requires complete action")
        if value.get("person_visibility") == "poor":
            errors.append("full label cannot have poor visibility")
        if value.get("third_person_present"):
            errors.append("full label cannot contain a persistent third person")
        if label.startswith(("playful_", "conflict_")) and person_count != 2:
            errors.append("interaction label requires exactly two primary people")
        if label.startswith("normal_") and person_count not in {1, 2}:
            errors.append("normal label requires one or two primary people")

    if label == "uncertain_intent":
        if motion not in {"chase", "push"}:
            errors.append("uncertain_intent requires chase or push")
        if decision not in {"accept_partial", "manual_review"}:
            errors.append("uncertain_intent requires partial or manual decision")

    if label in REJECT_LABELS and decision != "reject":
        errors.append("reject label requires reject decision")

    return ParsedReview(value, not errors, tuple(errors))


def parse_model_response(text: str) -> ParsedReview:
    try:
        return validate_review(extract_json_object(text))
    except (ValueError, json.JSONDecodeError) as exc:
        return ParsedReview({}, False, (str(exc),))


def _same_full(reviews: Sequence[ParsedReview]) -> Optional[str]:
    if not reviews or not all(review.valid for review in reviews):
        return None
    labels = {review.label for review in reviews}
    if len(labels) == 1:
        label = next(iter(labels))
        if label in CAMPUS6_LABELS:
            return label
    return None


def _same_partial(reviews: Sequence[ParsedReview]) -> Optional[str]:
    if not reviews or not all(review.valid for review in reviews):
        return None
    if all(review.label == "uncertain_intent" for review in reviews):
        motions = {review.motion for review in reviews}
        if len(motions) == 1 and next(iter(motions)) in {"chase", "push"}:
            return next(iter(motions))
    return None


def _same_reject(reviews: Sequence[ParsedReview]) -> Optional[str]:
    if not reviews or not all(review.valid for review in reviews):
        return None
    labels = {review.label for review in reviews}
    if len(labels) == 1 and next(iter(labels)) in REJECT_LABELS:
        return next(iter(labels))
    return None


def decide_reviews(
    independent: Sequence[ParsedReview],
    adjudicator: Optional[ParsedReview] = None,
    require_unanimous_interaction: bool = False,
) -> Dict[str, Any]:
    """Return a conservative final decision from two blind reviews and a tie-breaker."""

    full = _same_full(independent)
    if full:
        if require_unanimous_interaction and full.startswith(("playful_", "conflict_")):
            if adjudicator is None or not adjudicator.valid or adjudicator.label != full:
                return {
                    "status": "manual_review",
                    "reason": "pose_only_interaction_requires_three_model_consensus",
                }
            return {
                "status": "accepted",
                "label": full,
                "reason": "three_model_pose_interaction_consensus",
            }
        return {"status": "accepted", "label": full, "reason": "two_model_full_consensus"}

    partial = _same_partial(independent)
    if partial:
        return {"status": "partial", "motion": partial, "reason": "two_model_partial_consensus"}

    rejected = _same_reject(independent)
    if rejected:
        return {"status": "rejected", "label": rejected, "reason": "two_model_reject_consensus"}

    if adjudicator is not None and adjudicator.valid:
        for review in independent:
            if not review.valid:
                continue
            if review.label == adjudicator.label and review.label in CAMPUS6_LABELS:
                if require_unanimous_interaction and review.label.startswith(("playful_", "conflict_")):
                    continue
                return {
                    "status": "accepted",
                    "label": review.label,
                    "reason": "adjudicator_full_consensus",
                }
            if (
                review.label == adjudicator.label == "uncertain_intent"
                and review.motion == adjudicator.motion
            ):
                return {
                    "status": "partial",
                    "motion": review.motion,
                    "reason": "adjudicator_partial_consensus",
                }
            if review.label == adjudicator.label and review.label in REJECT_LABELS:
                return {
                    "status": "rejected",
                    "label": review.label,
                    "reason": "adjudicator_reject_consensus",
                }

    return {"status": "manual_review", "reason": "no_reliable_consensus"}


def reviews_need_adjudication(independent: Sequence[ParsedReview]) -> bool:
    return not (
        _same_full(independent)
        or _same_partial(independent)
        or _same_reject(independent)
    )
