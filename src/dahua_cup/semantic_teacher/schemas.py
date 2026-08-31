"""Strict, dependency-light schemas for teacher and pseudo-label records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


LABELS = (
    "normal_walk", "normal_run", "playful_chase",
    "playful_push", "conflict_chase", "conflict_push",
)


def normalize_distribution(
    value: Mapping[str, float] | Sequence[Mapping[str, Any]],
    labels: Sequence[str] = LABELS,
) -> dict[str, float]:
    allowed = tuple(str(label) for label in labels)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("labels must be non-empty and unique")
    if isinstance(value, Mapping):
        raw = {str(label): float(probability) for label, probability in value.items()}
    else:
        raw = {str(item["label"]): float(item["probability"]) for item in value}
    unsupported = sorted(set(raw) - set(allowed))
    if unsupported:
        raise ValueError("distribution contains unsupported labels: " + ", ".join(unsupported))
    distribution = {label: raw.get(label, 0.0) for label in allowed}
    if any(probability < 0 or probability > 1 for probability in distribution.values()):
        raise ValueError("probabilities must be in [0, 1]")
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("distribution must have positive mass")
    return {label: probability / total for label, probability in distribution.items()}


@dataclass(frozen=True)
class Evidence:
    type: str
    segment_id: str
    description: str


@dataclass
class TeacherOutput:
    sample_id: str
    label: str
    distribution: dict[str, float]
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    reason: str = ""
    needs_review: bool = False
    schema_version: str = "teacher_output.v1"

    def validate(self) -> None:
        if self.schema_version != "teacher_output.v1" or not self.sample_id:
            raise ValueError("invalid teacher schema version or sample_id")
        if self.label not in self.distribution:
            raise ValueError(f"unsupported label: {self.label}")
        self.distribution = normalize_distribution(
            self.distribution, tuple(self.distribution)
        )
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.label != max(self.distribution, key=self.distribution.get):
            raise ValueError("label must match distribution argmax")
        if abs(self.confidence - self.distribution[self.label]) > 0.05:
            raise ValueError("confidence must match the selected label probability")
        if not self.evidence:
            raise ValueError("teacher output requires at least one evidence item")
        if any(not item.segment_id for item in self.evidence):
            raise ValueError("all evidence must cite a segment")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        allowed_labels: Sequence[str] = LABELS,
    ) -> "TeacherOutput":
        data = dict(value)
        data["distribution"] = normalize_distribution(
            data.get("distribution", {}), allowed_labels
        )
        # Qwen may append measured fields such as ``pose_coverage`` to an
        # evidence object. They are useful in its private reasoning but are
        # not part of the stable teacher_output.v1 contract. Preserve the
        # required, auditable fields and ignore additive model-specific keys.
        data["evidence"] = [
            Evidence(**{
                key: item[key]
                for key in ("type", "segment_id", "description")
                if key in item
            })
            for item in data.get("evidence", [])
        ]
        result = cls(**data)
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["distribution"] = [
            {"label": label, "probability": probability}
            for label, probability in self.distribution.items()
        ]
        return data


@dataclass
class PseudoLabelRecord:
    sample_id: str
    label: str
    soft_label: list[float]
    quality_score: float
    status: str
    feature_version: str
    student_model_version: str
    teacher_model_version: str
    prompt_version: str
    filter_version: str
    source: str = "qwen_teacher"
    review: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "pseudo_label.v1"

    def validate(self) -> None:
        if self.schema_version != "pseudo_label.v1" or self.label not in LABELS:
            raise ValueError("invalid pseudo-label schema or label")
        if self.status not in {"accepted", "review", "rejected"}:
            raise ValueError("invalid pseudo-label status")
        if len(self.soft_label) != len(LABELS) or any(x < 0 for x in self.soft_label):
            raise ValueError("soft_label must contain six non-negative values")
        if abs(sum(self.soft_label) - 1.0) > 1e-5 or not 0 <= self.quality_score <= 1:
            raise ValueError("soft_label must sum to one and quality_score be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
