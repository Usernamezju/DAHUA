"""Fuse teacher/student signals and create versioned pseudo-label JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from dahua_cup.backend.store import ReviewStore
from dahua_cup.semantic_teacher.pseudo_label.calibration import (
    TemperatureCalibrator,
)
from dahua_cup.semantic_teacher.pseudo_label.filter_label import (
    FilterDecision,
    PseudoLabelFilter,
    load_filter_configuration,
)
from dahua_cup.semantic_teacher.schemas import (
    LABELS,
    TeacherOutput,
    normalize_distribution,
)

from .common import log_event, read_jsonl, write_jsonl


DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parents[1] / "configs" / "campus" / "thresholds.yaml"
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--student-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--review-db")
    parser.add_argument("--threshold-config", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--calibration")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _teacher_value(row: dict) -> dict:
    value = dict(row.get("result") or row)
    if not value.get("sample_id") and row.get("sample_id"):
        value["sample_id"] = row["sample_id"]
    return value


def _student_distribution(row: dict) -> dict[str, float]:
    value = row.get("distribution")
    if isinstance(value, list):
        value = {
            str(item["label"]): float(
                item.get("probability", item.get("score", 0.0))
            )
            for item in value
        }
    if value:
        return normalize_distribution(value)
    topk = row.get("topk") or row.get("top5") or []
    value = {
        str(item["label"]): float(item.get("score", item.get("probability", 0)))
        for item in topk
        if str(item.get("label")) in LABELS
    }
    if not value:
        raise ValueError(
            f"student sample {row.get('sample_id')} has no Campus6 distribution"
        )
    return normalize_distribution(value)


def _audit_selected(sample_id: str, rate: float, purpose: str) -> bool:
    digest = hashlib.sha256(f"{purpose}:{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < rate


def _load_calibrator(path: str | None) -> TemperatureCalibrator:
    if not path:
        return TemperatureCalibrator()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return TemperatureCalibrator.from_dict(value)


def main(argv=None):
    args = build_parser().parse_args(argv)
    thresholds, weights = load_filter_configuration(args.threshold_config)
    calibrator = _load_calibrator(args.calibration)
    teacher_rows = defaultdict(list)
    teacher_versions = {}
    for row in read_jsonl(args.teacher_jsonl):
        value = _teacher_value(row)
        output = TeacherOutput.from_dict(value)
        output.distribution = calibrator.transform(output.distribution)
        output.confidence = output.distribution[output.label]
        teacher_rows[output.sample_id].append(output)
        provenance = row.get("provenance") or {}
        teacher_versions[output.sample_id] = str(
            provenance.get("teacher_model_version")
            or row.get("model")
            or "Qwen3-VL-8B-Instruct"
        )
    students = {row["sample_id"]: row for row in read_jsonl(args.student_jsonl)}
    records = []
    review_items = []
    review_store = (
        ReviewStore(args.review_db)
        if args.review_db and not args.dry_run
        else None
    )
    filterer = PseudoLabelFilter(thresholds, weights)
    for sample_id, runs in sorted(teacher_rows.items()):
        student = students.get(sample_id)
        if student is None:
            log_event("pseudo_skip", sample_id=sample_id, reason="missing_student_output")
            continue
        decision = filterer.evaluate(
            runs,
            _student_distribution(student),
            float(student.get("pose_quality", 0.0)),
            float(student.get("temporal_stability", 0.0)),
            student.get("evidence_flags"),
            bool(student.get("rare_class", False))
            and _audit_selected(
                sample_id, thresholds.rare_class_audit_rate, "rare-class"
            ),
        )
        if decision.status == "accepted" and _audit_selected(
            sample_id, thresholds.accepted_audit_rate, "accepted-audit"
        ):
            decision = FilterDecision(
                "review",
                decision.score,
                decision.conflicts + ("accepted_random_audit",),
                decision.components,
            )
        versions = {
            "feature_version": student.get("feature_version", "feature.v1"),
            "student_model_version": student.get("model_version", "unknown"),
            "teacher_model_version": teacher_versions[sample_id],
            "prompt_version": args.prompt_version,
            "filter_version": "campus-filter-v1",
            "calibration_version": calibrator.version,
        }
        record = filterer.make_record(runs[0], decision, versions).to_dict()
        record["dataset_version"] = args.dataset_version
        record["calibration_version"] = calibrator.version
        record["filter_components"] = decision.components
        record["conflicts"] = list(decision.conflicts)
        record["confidence_method"] = (
            "temperature_scaling"
            if calibrator.version != "identity-v1"
            else "uncalibrated_teacher_distribution"
        )
        # Keep the immutable data references needed to inject an accepted
        # pseudo-label into a later training dataset.  These are copied from
        # the student/feature stage rather than inferred from the video name.
        for key in ("feature_path", "video_path", "source_hash"):
            if student.get(key):
                record[key] = student[key]
        records.append(record)
        if review_store and decision.status == "review":
            video_path = student.get("video_path")
            if not video_path:
                raise ValueError(
                    f"review sample {sample_id} requires student.video_path"
                )
            hard_score = float(student.get("hard_score", 0.0))
            review_items.append((record, video_path, hard_score))
    destination = Path(args.output_dir) / args.dataset_version / "pseudo_labels.jsonl"
    if not args.dry_run:
        write_jsonl(destination, records)
        for record, video_path, hard_score in review_items:
            review_store.enqueue_pseudo_record(
                record,
                video_path,
                pseudo_dataset_path=destination,
                hard_score=hard_score,
            )
    log_event("pseudo_complete", samples=len(records), statuses={status: sum(r["status"] == status for r in records)
              for status in ("accepted", "review", "rejected")}, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
