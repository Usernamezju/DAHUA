"""Dependency-light calibration for closed-set teacher distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from dahua_cup.semantic_teacher.schemas import LABELS, normalize_distribution


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Calibrate class probabilities with a validation-set fitted temperature."""

    temperature: float = 1.0
    version: str = "identity-v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be positive")
        if not self.version:
            raise ValueError("calibration version is required")

    def transform(
        self,
        distribution: Mapping[str, float],
        labels: Sequence[str] = LABELS,
    ) -> dict[str, float]:
        probability = normalize_distribution(distribution, labels)
        power = 1.0 / self.temperature
        adjusted = {
            label: max(probability[label], 1e-12) ** power for label in labels
        }
        total = sum(adjusted.values())
        return {label: adjusted[label] / total for label in labels}

    def to_dict(self) -> dict:
        return {
            "schema_version": "teacher_calibration.v1",
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "TemperatureCalibrator":
        if value.get("schema_version") != "teacher_calibration.v1":
            raise ValueError("unsupported teacher calibration schema")
        return cls(
            temperature=float(value["temperature"]),
            version=str(value["version"]),
        )


def negative_log_likelihood(
    samples: Iterable[tuple[Mapping[str, float], str]],
    temperature: float,
    labels: Sequence[str] = LABELS,
) -> float:
    calibrator = TemperatureCalibrator(temperature)
    values = list(samples)
    if not values:
        raise ValueError("calibration requires at least one labeled sample")
    loss = 0.0
    allowed = set(labels)
    for distribution, true_label in values:
        if true_label not in allowed:
            raise ValueError(f"unsupported calibration label: {true_label}")
        probability = calibrator.transform(distribution, labels)
        loss -= math.log(max(probability[true_label], 1e-12))
    return loss / len(values)


def fit_temperature(
    samples: Iterable[tuple[Mapping[str, float], str]],
    *,
    labels: Sequence[str] = LABELS,
    minimum: float = 0.25,
    maximum: float = 5.0,
    steps: int = 192,
    version: str = "campus6-temperature-v1",
) -> TemperatureCalibrator:
    """Fit temperature by deterministic grid search, avoiding scipy/sklearn."""
    values = list(samples)
    if steps < 2 or minimum <= 0 or maximum <= minimum:
        raise ValueError("invalid temperature search interval")
    candidates = [
        minimum + (maximum - minimum) * index / (steps - 1)
        for index in range(steps)
    ]
    best = min(
        candidates,
        key=lambda temperature: negative_log_likelihood(
            values, temperature, labels
        ),
    )
    return TemperatureCalibrator(best, version)
