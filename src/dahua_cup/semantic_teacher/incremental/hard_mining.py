"""Model-agnostic hard-sample scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


BEFORE_TEACHER_WEIGHTS = {
    "student_entropy": 0.35,
    "margin_uncertainty": 0.20,
    "temporal_instability": 0.15,
    "quality_failure": 0.10,
    "novelty": 0.10,
    "rare_class": 0.10,
}
AFTER_TEACHER_WEIGHTS = {
    "student_entropy": 0.25,
    "margin_uncertainty": 0.10,
    "student_teacher_disagreement": 0.25,
    "temporal_instability": 0.10,
    "quality_failure": 0.05,
    "novelty": 0.05,
    "rare_class": 0.10,
    "rule_conflict": 0.10,
}

def _raw_probabilities(
    distribution: Mapping[str, float],
) -> dict[str, float]:
    if not distribution:
        raise ValueError("distribution must not be empty")
    values = {str(label): float(value) for label, value in distribution.items()}
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in values.values()
    ):
        raise ValueError("distribution probabilities must be finite and in [0, 1]")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("distribution must have positive mass")
    if total > 1.0 + 1e-5:
        values = {label: value / total for label, value in values.items()}
    return values


def _entropy_values(
    distribution: Mapping[str, float], label_space_size: int
) -> list[float]:
    values = list(_raw_probabilities(distribution).values())
    if label_space_size < len(values) or label_space_size < 2:
        raise ValueError(
            "label_space_size must cover the observed labels and be at least two"
        )
    total = sum(values)
    if total > 1:
        values = [value / total for value in values]
        total = 1.0
    missing_classes = label_space_size - len(values)
    residual = max(0.0, 1.0 - total)
    if missing_classes and residual:
        values.extend([residual / missing_classes] * missing_classes)
    elif total and abs(total - 1.0) > 1e-8:
        values = [value / total for value in values]
    return values


def normalized_entropy(
    distribution: Mapping[str, float],
    label_space_size: int | None = None,
) -> float:
    size = int(label_space_size or max(2, len(distribution)))
    values = _entropy_values(distribution, size)
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    return max(0.0, min(1.0, entropy / math.log(size)))


def margin_uncertainty(distribution: Mapping[str, float]) -> float:
    """Return one minus the top-1/top-2 probability margin."""
    values = sorted(_raw_probabilities(distribution).values(), reverse=True)
    if len(values) < 2:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (values[0] - values[1])))


def distribution_similarity(
    left: Mapping[str, float],
    right: Mapping[str, float],
    label_space_size: int | None = None,
) -> float:
    """Jensen-Shannon similarity for full or Top-k class distributions."""
    p = _raw_probabilities(left)
    q = _raw_probabilities(right)
    labels = sorted(set(p) | set(q))
    size = int(label_space_size or max(2, len(labels)))
    if size < len(labels):
        raise ValueError("label_space_size is smaller than observed labels")
    p_values = [p.get(label, 0.0) for label in labels]
    q_values = [q.get(label, 0.0) for label in labels]
    missing_classes = size - len(labels)
    p_residual = max(0.0, 1.0 - sum(p_values))
    q_residual = max(0.0, 1.0 - sum(q_values))
    if missing_classes:
        p_values.append(p_residual)
        q_values.append(q_residual)
    else:
        p_total = sum(p_values)
        q_total = sum(q_values)
        p_values = [value / p_total for value in p_values]
        q_values = [value / q_total for value in q_values]
    midpoint = [
        (left_value + right_value) / 2
        for left_value, right_value in zip(p_values, q_values)
    ]

    def kl(values, reference):
        return sum(
            value * math.log(value / target)
            for value, target in zip(values, reference)
            if value > 0 and target > 0
        )

    jsd = (kl(p_values, midpoint) + kl(q_values, midpoint)) / 2
    return max(0.0, min(1.0, 1.0 - jsd / math.log(2.0)))


@dataclass(frozen=True)
class HardMiningDecision:
    score: float
    components: dict[str, float]
    reasons: tuple[str, ...]


def hard_sample_score(
    student,
    teacher=None,
    temporal_instability=0.0,
    quality_failure=0.0,
    label_space_size=None,
) -> float:
    entropy = normalized_entropy(student, label_space_size)
    disagreement = (
        0.0
        if teacher is None
        else 1.0 - distribution_similarity(
            student, teacher, label_space_size
        )
    )
    score = 0.45 * entropy + 0.30 * disagreement + 0.15 * temporal_instability + 0.10 * quality_failure
    return max(0.0, min(1.0, float(score)))


def evaluate_hard_sample(
    student: Mapping[str, float],
    *,
    teacher: Mapping[str, float] | None = None,
    temporal_instability: float = 0.0,
    quality_failure: float = 0.0,
    novelty: float = 0.0,
    rare_class: float = 0.0,
    rule_conflict: float = 0.0,
    label_space_size: int | None = None,
    before_teacher_weights: Mapping[str, float] | None = None,
    after_teacher_weights: Mapping[str, float] | None = None,
    reason_threshold: float = 0.60,
) -> HardMiningDecision:
    """Score information value before or after teacher inference.

    Before teacher inference, entropy and margin uncertainty dominate selection.
    After teacher inference, distribution disagreement receives a larger share.
    """
    clamp = lambda value: max(0.0, min(1.0, float(value)))
    components = {
        "student_entropy": normalized_entropy(student, label_space_size),
        "margin_uncertainty": margin_uncertainty(student),
        "temporal_instability": clamp(temporal_instability),
        "quality_failure": clamp(quality_failure),
        "novelty": clamp(novelty),
        "rare_class": clamp(rare_class),
        "rule_conflict": clamp(rule_conflict),
    }
    if not 0 <= reason_threshold <= 1:
        raise ValueError("reason_threshold must be in [0, 1]")
    if teacher is None:
        weights = dict(before_teacher_weights or BEFORE_TEACHER_WEIGHTS)
        expected = set(BEFORE_TEACHER_WEIGHTS)
    else:
        components["student_teacher_disagreement"] = (
            1.0 - distribution_similarity(
                student, teacher, label_space_size
            )
        )
        weights = dict(after_teacher_weights or AFTER_TEACHER_WEIGHTS)
        expected = set(AFTER_TEACHER_WEIGHTS)
    if set(weights) != expected:
        raise ValueError(
            "hard-mining weights must contain exactly: "
            + ", ".join(sorted(expected))
        )
    if (
        any(not 0 <= float(value) <= 1 for value in weights.values())
        or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6
    ):
        raise ValueError("hard-mining weights must be in [0, 1] and sum to one")
    score = sum(components[name] * weight for name, weight in weights.items())
    reasons = tuple(
        name
        for name, value in sorted(components.items())
        if value >= reason_threshold
    )
    return HardMiningDecision(clamp(score), components, reasons)
