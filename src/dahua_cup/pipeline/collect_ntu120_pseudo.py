"""Collect versioned NTU120 pseudo labels from Web student/teacher artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from dahua_cup.pipeline.common import file_hash, write_jsonl


@dataclass(frozen=True)
class NTU120PseudoThresholds:
    accept_score: float = 0.78
    review_score: float = 0.50
    minimum_teacher_confidence: float = 0.60
    minimum_pose_quality: float = 0.70

    def validate(self) -> None:
        values = (
            self.accept_score,
            self.review_score,
            self.minimum_teacher_confidence,
            self.minimum_pose_quality,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("NTU120 pseudo thresholds must be in [0, 1]")
        if self.review_score > self.accept_score:
            raise ValueError("review_score must not exceed accept_score")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accept-score", type=float, default=0.78)
    parser.add_argument("--review-score", type=float, default=0.50)
    parser.add_argument("--minimum-teacher-confidence", type=float, default=0.60)
    parser.add_argument("--minimum-pose-quality", type=float, default=0.70)
    return parser


def load_labels(path: str | Path) -> tuple[str, ...]:
    labels = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(labels) not in {6, 120} or len(set(labels)) != len(labels):
        raise ValueError("label map must contain either 6 Campus6 or 120 NTU labels")
    return labels


def _distribution(value, labels: Sequence[str]) -> dict[str, float]:
    if isinstance(value, list):
        raw = {
            str(item["label"]): float(
                item.get("probability", item.get("score", 0.0))
            )
            for item in value
        }
    elif isinstance(value, Mapping):
        raw = {str(label): float(score) for label, score in value.items()}
    else:
        raise ValueError("teacher distribution must be a mapping or list")
    allowed = set(labels)
    if not raw or set(raw) - allowed:
        raise ValueError("teacher distribution contains unsupported labels")
    if any(not math.isfinite(score) or score < 0 for score in raw.values()):
        raise ValueError("teacher distribution contains an invalid probability")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("teacher distribution has zero probability mass")
    return {label: raw.get(label, 0.0) / total for label in labels}


def _teacher_result(value: dict) -> dict:
    return dict(value.get("result") or value)


def _pose_quality(prediction: dict) -> float:
    gate = dict(prediction.get("teacher_gate") or {})
    metrics = dict(gate.get("pose_metrics") or {})
    value = metrics.get("quality", gate.get("pose_quality", 0.0))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _stability(prediction: dict) -> float:
    gate = dict(prediction.get("teacher_gate") or {})
    instability = dict(gate.get("instability") or {})
    try:
        score = float(instability.get("score", 0.0))
    except (TypeError, ValueError):
        score = 1.0
    return 1.0 - max(0.0, min(1.0, score))


def make_record(
    prediction: dict,
    teacher_payload: dict,
    labels: Sequence[str],
    thresholds: NTU120PseudoThresholds,
) -> dict:
    thresholds.validate()
    result = _teacher_result(teacher_payload)
    sample_id = str(
        result.get("sample_id")
        or teacher_payload.get("sample_id")
        or prediction.get("sample_id")
        or ""
    )
    if not sample_id:
        raise ValueError("pseudo sample has no sample_id")
    label = str(result.get("label", ""))
    if label not in labels:
        raise ValueError(f"teacher label is outside configured label map: {label}")
    distribution = _distribution(result.get("distribution"), labels)
    teacher_confidence = float(result.get("confidence", distribution[label]))
    if not math.isfinite(teacher_confidence) or not 0 <= teacher_confidence <= 1:
        raise ValueError("teacher confidence must be finite and in [0, 1]")
    topk = list(prediction.get("topk") or [])
    student_label = str(topk[0].get("label", "")) if topk else ""
    agreement = 1.0 if student_label == label else 0.0
    pose_quality = _pose_quality(prediction)
    stability = _stability(prediction)
    gate = dict(prediction.get("teacher_gate") or {})
    conflict = dict(gate.get("teacher_conflict") or {})
    needs_review = bool(result.get("needs_review")) or bool(
        conflict.get("conflict")
    )
    components = {
        "teacher_confidence": teacher_confidence,
        "student_teacher_label_agreement": agreement,
        "pose_quality": pose_quality,
        "temporal_stability": stability,
    }
    score = (
        0.45 * teacher_confidence
        + 0.20 * agreement
        + 0.20 * pose_quality
        + 0.15 * stability
    )
    reasons = []
    quality_warnings = []
    feature_path = Path(str(prediction.get("feature", ""))).expanduser()
    if not feature_path.is_file():
        reasons.append("pose_feature_missing")
    if teacher_confidence < thresholds.minimum_teacher_confidence:
        reasons.append("teacher_confidence_below_threshold")
    if pose_quality < thresholds.minimum_pose_quality:
        # Pose quality already contributes continuously to the combined
        # score.  Keep it auditable without turning one weak signal into an
        # automatic review decision.
        quality_warnings.append("pose_quality_below_threshold")
    if result.get("needs_review"):
        reasons.append("teacher_requested_review")
    if conflict.get("conflict"):
        reasons.append("high_confidence_student_teacher_conflict")
    if (
        not reasons
        and not needs_review
        and score >= thresholds.accept_score
    ):
        status = "accepted"
    elif needs_review or score >= thresholds.review_score:
        status = "review"
    else:
        status = "rejected"
    if not reasons and status != "accepted":
        reasons.append("combined_quality_below_acceptance")
    vector = [distribution[label_name] for label_name in labels]
    teacher_path = teacher_payload.get("_artifact_path", "")
    prediction_path = prediction.get("_artifact_path", "")
    fingerprint_payload = {
        "sample_id": sample_id,
        "teacher_hash": (
            file_hash(teacher_path) if teacher_path else ""
        ),
        "prediction_hash": (
            file_hash(prediction_path) if prediction_path else ""
        ),
        "thresholds": thresholds.__dict__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "campus6_pseudo_label.v1" if len(labels) == 6 else "ntu120_pseudo_label.v1",
        "sample_id": sample_id,
        "task": "campus6_rtmpose17" if len(labels) == 6 else "ntu120_xsub",
        "label_space_size": len(labels),
        "label": label,
        "label_index": list(labels).index(label),
        "soft_label": vector,
        "quality_score": max(0.0, min(1.0, score)),
        "status": status,
        "review_reasons": reasons,
        "quality_warnings": quality_warnings,
        "filter_components": components,
        "teacher_confidence": teacher_confidence,
        "confidence_method": "qwen_self_report_uncalibrated",
        "student_top1_label": student_label,
        "feature_path": str(feature_path.resolve()) if feature_path.is_file() else "",
        "student_checkpoint": prediction.get("checkpoint", ""),
        "student_checkpoint_sha256": prediction.get("checkpoint_sha256", ""),
        "teacher_model": teacher_payload.get("model", ""),
        "teacher_artifact": teacher_path,
        "prediction_artifact": prediction_path,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_records(
    prediction_dir: str | Path,
    teacher_dir: str | Path,
    labels: Sequence[str],
    thresholds: NTU120PseudoThresholds | None = None,
) -> list[dict]:
    thresholds = thresholds or NTU120PseudoThresholds()
    predictions = {}
    for path in sorted(Path(prediction_dir).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        task = str(value.get("task", ""))
        try:
            label_space_size = int(value.get("label_space_size", 0))
        except (TypeError, ValueError):
            label_space_size = 0
        if label_space_size != len(labels) and not (len(labels) == 120 and task.startswith("ntu120")):
            continue
        if value.get("status", "completed") != "completed":
            continue
        value["_artifact_path"] = str(path.resolve())
        predictions[str(value.get("sample_id") or path.stem)] = value
    records = []
    for path in sorted(Path(teacher_dir).glob("*.json")):
        teacher = json.loads(path.read_text(encoding="utf-8"))
        teacher["_artifact_path"] = str(path.resolve())
        result = _teacher_result(teacher)
        sample_id = str(result.get("sample_id") or path.stem)
        prediction = predictions.get(sample_id)
        if prediction is None:
            continue
        records.append(make_record(prediction, teacher, labels, thresholds))
    return sorted(records, key=lambda row: row["sample_id"])


def merge_records(existing: Sequence[dict], current: Sequence[dict]) -> list[dict]:
    merged = {str(row["sample_id"]): dict(row) for row in existing}
    for row in current:
        previous = merged.get(str(row["sample_id"]))
        if previous and previous.get("fingerprint") == row.get("fingerprint"):
            continue
        value = dict(row)
        value["version"] = int(previous.get("version", 0)) + 1 if previous else 1
        value["supersedes_fingerprint"] = (
            str(previous.get("fingerprint", "")) if previous else ""
        )
        merged[str(row["sample_id"])] = value
    return [merged[sample_id] for sample_id in sorted(merged)]


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    labels = load_labels(args.label_map)
    thresholds = NTU120PseudoThresholds(
        accept_score=args.accept_score,
        review_score=args.review_score,
        minimum_teacher_confidence=args.minimum_teacher_confidence,
        minimum_pose_quality=args.minimum_pose_quality,
    )
    destination = Path(args.output)
    existing = []
    if destination.is_file():
        existing = [
            json.loads(line)
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    records = merge_records(
        existing,
        collect_records(
            args.prediction_dir,
            args.teacher_dir,
            labels,
            thresholds,
        ),
    )
    write_jsonl(destination, records)
    counts = {
        status: sum(row["status"] == status for row in records)
        for status in ("accepted", "review", "rejected")
    }
    print(json.dumps({"samples": len(records), "statuses": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
