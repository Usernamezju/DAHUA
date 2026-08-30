"""Versioned feature artifact validation and serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FEATURE_SCHEMA_VERSION = "feature.v1"


@dataclass
class QualityMetrics:
    pose_coverage: float
    mean_joint_score: float
    track_fragmentation: float = 0.0
    id_switch_estimate: int = 0

    def validate(self) -> None:
        for name in ("pose_coverage", "mean_joint_score", "track_fragmentation"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality.{name} must be in [0, 1]")
        if self.id_switch_estimate < 0:
            raise ValueError("quality.id_switch_estimate must be non-negative")


@dataclass
class FeatureMetadata:
    sample_id: str
    source_hash: str
    fps: float
    width: int
    height: int
    start_ms: int
    end_ms: int
    extractor_versions: dict[str, str] = field(default_factory=dict)
    quality: QualityMetrics = field(default_factory=lambda: QualityMetrics(0.0, 0.0))
    schema_version: str = FEATURE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema: {self.schema_version}")
        if not self.sample_id or not self.source_hash:
            raise ValueError("sample_id and source_hash are required")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("fps, width and height must be positive")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid clip time range")
        self.quality.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureMetadata":
        data = dict(value)
        data["quality"] = QualityMetrics(**data.get("quality", {}))
        result = cls(**data)
        result.validate()
        return result


REQUIRED_ARRAYS = {
    "keypoint": (4, np.float32),
    "keypoint_score": (3, np.float32),
    "bbox": (3, np.float32),
    "track_id": (1, np.int64),
    "valid_mask": (3, np.bool_),
    "center": (3, np.float32),
    "velocity": (3, np.float32),
    "acceleration": (3, np.float32),
}


def validate_feature_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the feature.v1 NPZ contract and return normalized arrays."""
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise ValueError(f"missing feature arrays: {', '.join(missing)}")
    normalized: dict[str, np.ndarray] = {}
    for name, (ndim, dtype) in REQUIRED_ARRAYS.items():
        value = np.asarray(arrays[name])
        if value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions, got {value.shape}")
        normalized[name] = value.astype(dtype, copy=False)

    keypoint = normalized["keypoint"]
    if keypoint.shape[-2:] != (17, 2):
        raise ValueError("keypoint must have shape [M, T, 17, 2]")
    m, t = keypoint.shape[:2]
    expected = {
        "keypoint_score": (m, t, 17), "valid_mask": (m, t, 17),
        "bbox": (m, t, 4), "track_id": (m,), "center": (m, t, 2),
        "velocity": (m, t, 2), "acceleration": (m, t, 2),
    }
    for name, shape in expected.items():
        if normalized[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {normalized[name].shape}")
    if not np.isfinite(normalized["keypoint"]).all():
        raise ValueError("keypoint contains non-finite values")
    return normalized


def save_feature_artifact(
    output_stem: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: FeatureMetadata,
) -> tuple[Path, Path]:
    """Atomically-ish write a matching compressed NPZ and JSON metadata pair."""
    normalized = validate_feature_arrays(arrays)
    metadata.validate()
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
    np.savez_compressed(npz_path, **normalized)
    json_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path


def stable_sample_id(source_hash: str, start_ms: int, end_ms: int) -> str:
    payload = f"{source_hash}:{int(start_ms)}:{int(end_ms)}".encode()
    return hashlib.sha256(payload).hexdigest()
