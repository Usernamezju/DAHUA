"""Multi-signal pseudo-label scoring and deterministic routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dahua_cup.semantic_teacher.schemas import (
    LABELS,
    PseudoLabelRecord,
    TeacherOutput,
    normalize_distribution,
)


def distribution_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return 1 - normalized Jensen-Shannon divergence in [0, 1]."""
    p, q = normalize_distribution(left), normalize_distribution(right)
    midpoint = {label: (p[label] + q[label]) / 2 for label in LABELS}

    def kl(a, b):
        return sum(a[label] * math.log(a[label] / b[label]) for label in LABELS if a[label] > 0)

    jsd = (kl(p, midpoint) + kl(q, midpoint)) / 2
    return max(0.0, min(1.0, 1.0 - jsd / math.log(2.0)))


@dataclass(frozen=True)
class FilterThresholds:
    accept_score: float = 0.85
    review_score: float = 0.55
    minimum_pose_coverage: float = 0.70
    high_confidence: float = 0.80
    accepted_audit_rate: float = 0.10
    rare_class_audit_rate: float = 0.20

    def validate(self) -> None:
        values = (
            self.accept_score,
            self.review_score,
            self.minimum_pose_coverage,
            self.high_confidence,
            self.accepted_audit_rate,
            self.rare_class_audit_rate,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("all filter thresholds and audit rates must be in [0, 1]")
        if self.review_score > self.accept_score:
            raise ValueError("review_score must not exceed accept_score")

    @classmethod
    def from_mapping(cls, value: Mapping) -> "FilterThresholds":
        pseudo = dict(value.get("pseudo_label", value))
        pose = dict(value.get("pose", {}))
        result = cls(
            accept_score=float(pseudo.get("accept_score", cls.accept_score)),
            review_score=float(pseudo.get("review_score", cls.review_score)),
            minimum_pose_coverage=float(
                pseudo.get(
                    "minimum_pose_coverage",
                    pose.get("minimum_coverage", cls.minimum_pose_coverage),
                )
            ),
            high_confidence=float(
                pseudo.get("high_confidence", cls.high_confidence)
            ),
            accepted_audit_rate=float(
                pseudo.get("accepted_audit_rate", cls.accepted_audit_rate)
            ),
            rare_class_audit_rate=float(
                pseudo.get("rare_class_audit_rate", cls.rare_class_audit_rate)
            ),
        )
        result.validate()
        return result


def load_filter_configuration(
    path: str | Path | None,
) -> tuple[FilterThresholds, dict[str, float]]:
    """Load production filter thresholds and weights from the campus YAML."""
    if path is None:
        thresholds = FilterThresholds()
        thresholds.validate()
        return thresholds, dict(PseudoLabelFilter.WEIGHTS)
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pseudo-label threshold config not found: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    thresholds = FilterThresholds.from_mapping(value)
    pseudo = dict(value.get("pseudo_label", value))
    weights = {
        str(name): float(weight)
        for name, weight in dict(pseudo.get("weights", PseudoLabelFilter.WEIGHTS)).items()
    }
    return thresholds, weights


@dataclass(frozen=True)
class FilterDecision:
    status: str
    score: float
    conflicts: tuple[str, ...]
    components: dict[str, float]


class PseudoLabelFilter:
    WEIGHTS = {
        "llm_consistency": 0.30,
        "teacher_probability": 0.25,
        "student_teacher_agreement": 0.20,
        "pose_quality": 0.15,
        "temporal_stability": 0.10,
    }

    def __init__(self, thresholds: FilterThresholds | None = None, weights=None):
        self.thresholds = thresholds or FilterThresholds()
        self.thresholds.validate()
        self.weights = dict(weights or self.WEIGHTS)
        if set(self.weights) != set(self.WEIGHTS) or abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("filter weights must contain the five components and sum to one")

    def evaluate(
        self,
        teacher_runs: Sequence[TeacherOutput],
        student_distribution: Mapping[str, float],
        pose_quality: float,
        temporal_stability: float,
        evidence_flags: Mapping[str, bool] | None = None,
        rare_class: bool = False,
    ) -> FilterDecision:
        if not teacher_runs:
            return FilterDecision("rejected", 0.0, ("missing_teacher_output",), {})
        for run in teacher_runs:
            run.validate()
        teacher = teacher_runs[0]
        consistency = min(
            (distribution_similarity(teacher.distribution, run.distribution) for run in teacher_runs[1:]),
            default=1.0,
        )
        components = {
            "llm_consistency": consistency,
            "teacher_probability": teacher.distribution[teacher.label],
            "student_teacher_agreement": distribution_similarity(teacher.distribution, student_distribution),
            "pose_quality": max(0.0, min(1.0, float(pose_quality))),
            "temporal_stability": max(0.0, min(1.0, float(temporal_stability))),
        }
        score = sum(self.weights[name] * value for name, value in components.items())
        flags = dict(evidence_flags or {})
        conflicts: list[str] = []
        student = normalize_distribution(student_distribution)
        student_label = max(student, key=student.get)
        if (
            student_label != teacher.label
            and student[student_label] >= self.thresholds.high_confidence
            and teacher.confidence >= self.thresholds.high_confidence
        ):
            conflicts.append("high_confidence_student_teacher_disagreement")
        if teacher.label.endswith("push") and not flags.get("close_contact", False) and pose_quality >= 0.7:
            conflicts.append("push_without_contact_evidence")
        if teacher.label.endswith("chase") and not flags.get("sustained_relative_motion", False):
            conflicts.append("chase_without_sustained_motion")
        if len({run.label for run in teacher_runs}) > 1:
            conflicts.append("teacher_run_disagreement")
        if flags.get("tracking_failure", False):
            conflicts.append("tracking_or_pose_failure")
        if rare_class:
            conflicts.append("rare_class_audit")

        if pose_quality < self.thresholds.minimum_pose_coverage:
            status = "rejected" if score < self.thresholds.review_score else "review"
        elif conflicts or teacher.needs_review:
            status = "review"
        elif score >= self.thresholds.accept_score:
            status = "accepted"
        elif score >= self.thresholds.review_score:
            status = "review"
        else:
            status = "rejected"
        return FilterDecision(status, score, tuple(conflicts), components)

    @staticmethod
    def make_record(
        teacher: TeacherOutput,
        decision: FilterDecision,
        versions: Mapping[str, str],
    ) -> PseudoLabelRecord:
        return PseudoLabelRecord(
            sample_id=teacher.sample_id,
            label=teacher.label,
            soft_label=[teacher.distribution[label] for label in LABELS],
            quality_score=decision.score,
            status=decision.status,
            feature_version=versions["feature_version"],
            student_model_version=versions["student_model_version"],
            teacher_model_version=versions["teacher_model_version"],
            prompt_version=versions["prompt_version"],
            filter_version=versions["filter_version"],
        )
