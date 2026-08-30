"""Rank arbitrary-label student predictions by information value."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from dahua_cup.backend.store import ReviewStore
from dahua_cup.pipeline.common import log_event, read_jsonl, write_jsonl
from dahua_cup.semantic_teacher.incremental.hard_mining import (
    AFTER_TEACHER_WEIGHTS,
    BEFORE_TEACHER_WEIGHTS,
    evaluate_hard_sample,
)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "campus"
    / "hard_mining.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    students = parser.add_mutually_exclusive_group(required=True)
    students.add_argument("--student-jsonl")
    students.add_argument(
        "--prediction-dir",
        help="Web artifacts/predictions directory containing per-sample JSON",
    )
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output")
    parser.add_argument("--review-db")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-class-limit", type=int)
    parser.add_argument("--minimum-score", type=float)
    parser.add_argument("--rare-class-quantile", type=float)
    parser.add_argument("--reason-threshold", type=float)
    parser.add_argument("--label-space-size", type=int)
    return parser


def load_mining_configuration(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"hard-mining config not found: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != "hard_mining.v1":
        raise ValueError("unsupported hard-mining config schema")
    selection = dict(value.get("selection") or {})
    weights = dict(value.get("weights") or {})
    result = {
        "minimum_score": float(selection.get("minimum_score", 0.0)),
        "limit": int(selection.get("limit", 0)),
        "per_class_limit": int(selection.get("per_class_limit", 0)),
        "rare_class_quantile": float(
            selection.get("rare_class_quantile", 0.25)
        ),
        "reason_threshold": float(selection.get("reason_threshold", 0.60)),
        "before_teacher_weights": dict(
            weights.get("before_teacher") or BEFORE_TEACHER_WEIGHTS
        ),
        "after_teacher_weights": dict(
            weights.get("after_teacher") or AFTER_TEACHER_WEIGHTS
        ),
    }
    if result["limit"] < 0 or result["per_class_limit"] < 0:
        raise ValueError("configured candidate budgets must be non-negative")
    for name in (
        "minimum_score",
        "rare_class_quantile",
        "reason_threshold",
    ):
        if not 0 <= result[name] <= 1:
            raise ValueError(f"configured {name} must be in [0, 1]")
    return result


def _distribution(row: dict) -> dict[str, float]:
    value = row.get("distribution")
    if value is None:
        value = row.get("topk") or row.get("top5")
    if isinstance(value, list):
        value = {
            str(item["label"]): float(item.get("score", item.get("probability", 0)))
            for item in value
        }
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"sample {row.get('sample_id')} has no class distribution"
        )
    result = {str(label): float(score) for label, score in value.items()}
    if any(score < 0 or score > 1 for score in result.values()):
        raise ValueError(
            f"sample {row.get('sample_id')} has an invalid probability"
        )
    if sum(result.values()) <= 0:
        raise ValueError(
            f"sample {row.get('sample_id')} has zero probability mass"
        )
    return result


def _label_space_size(
    row: dict, distribution: dict[str, float], override: int | None
) -> int:
    if override is not None:
        size = int(override)
    elif row.get("label_space_size"):
        size = int(row["label_space_size"])
    else:
        size = len(distribution)
    if size < max(2, len(distribution)):
        raise ValueError(
            f"invalid label_space_size for sample {row.get('sample_id')}: {size}"
        )
    return size


def class_statistics(
    predicted_labels: list[str], rare_class_quantile: float = 0.25
) -> dict:
    if not 0 <= rare_class_quantile <= 1:
        raise ValueError("rare_class_quantile must be in [0, 1]")
    counts = Counter(predicted_labels)
    if not counts:
        return {
            "sample_count": 0,
            "class_counts": {},
            "rare_count_cutoff": 0,
            "rare_labels": [],
        }
    ordered_counts = sorted(counts.values())
    index = int((len(ordered_counts) - 1) * rare_class_quantile)
    cutoff = ordered_counts[index]
    maximum = max(ordered_counts)
    rare_labels = sorted(
        label
        for label, count in counts.items()
        if count <= cutoff and count < maximum
    )
    return {
        "sample_count": len(predicted_labels),
        "observed_class_count": len(counts),
        "class_counts": dict(sorted(counts.items())),
        "rare_class_quantile": rare_class_quantile,
        "rare_count_cutoff": cutoff,
        "rare_labels": rare_labels,
    }


def mine(
    rows: list[dict],
    teacher_rows: dict[str, dict] | None = None,
    *,
    rare_class_quantile: float = 0.25,
    label_space_size: int | None = None,
    before_teacher_weights: dict[str, float] | None = None,
    after_teacher_weights: dict[str, float] | None = None,
    reason_threshold: float = 0.60,
) -> list[dict]:
    prepared = []
    teacher_rows = teacher_rows or {}
    for row in rows:
        sample_id = str(row["sample_id"])
        student = _distribution(row)
        predicted_label = max(student, key=student.get)
        prepared.append(
            (
                row,
                sample_id,
                student,
                predicted_label,
                _label_space_size(row, student, label_space_size),
            )
        )
    statistics = class_statistics(
        [item[3] for item in prepared], rare_class_quantile
    )
    counts = statistics["class_counts"]
    maximum_count = max(counts.values(), default=1)
    rare_labels = set(statistics["rare_labels"])

    result = []
    for row, sample_id, student, predicted_label, space_size in prepared:
        teacher_row = teacher_rows.get(sample_id)
        teacher_value = (
            teacher_row.get("result") or teacher_row
            if teacher_row else None
        )
        teacher = _distribution(teacher_value) if teacher_value else None
        gate = dict(row.get("teacher_gate") or {})
        pose_metrics = dict(gate.get("pose_metrics") or {})
        instability = dict(gate.get("instability") or {})
        conflict = dict(gate.get("teacher_conflict") or {})
        quality_failure = row.get("quality_failure")
        if quality_failure is None:
            pose_quality = row.get(
                "pose_quality",
                pose_metrics.get("quality", gate.get("pose_quality", 1.0)),
            )
            quality_failure = 1.0 - float(pose_quality)
        temporal_instability = row.get(
            "temporal_instability", instability.get("score", 0.0)
        )
        rule_conflict = row.get(
            "rule_conflict", float(bool(conflict.get("conflict", False)))
        )
        class_count = counts[predicted_label]
        rarity_score = (
            0.0
            if maximum_count <= 1
            else 1.0 - class_count / maximum_count
        )
        automatic_rare = predicted_label in rare_labels
        rare_score = max(
            rarity_score,
            float(bool(row.get("rare_class", False))),
        )
        decision = evaluate_hard_sample(
            student,
            teacher=teacher,
            temporal_instability=float(temporal_instability),
            quality_failure=float(quality_failure),
            novelty=float(row.get("novelty", 0.0)),
            rare_class=rare_score,
            rule_conflict=float(rule_conflict),
            label_space_size=space_size,
            before_teacher_weights=before_teacher_weights,
            after_teacher_weights=after_teacher_weights,
            reason_threshold=reason_threshold,
        )
        result.append(
            {
                **row,
                "label_space_size": space_size,
                "predicted_label": predicted_label,
                "predicted_class_count": class_count,
                "predicted_class_frequency": (
                    class_count / max(1, statistics["sample_count"])
                ),
                "rarity_score": rarity_score,
                "rare_class": (
                    automatic_rare or bool(row.get("rare_class", False))
                ),
                "hard_score": decision.score,
                "review_priority": int(round(decision.score * 1000)),
                "hard_components": decision.components,
                "hard_reasons": list(decision.reasons),
            }
        )
    return sorted(result, key=lambda item: (-item["hard_score"], item["sample_id"]))


def read_prediction_directory(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {source}")
    rows = []
    for prediction_path in sorted(source.glob("*.json")):
        value = json.loads(prediction_path.read_text(encoding="utf-8"))
        value.setdefault("sample_id", prediction_path.stem)
        rows.append(value)
    if not rows:
        raise ValueError(f"prediction directory contains no JSON files: {source}")
    return rows


def select_budget(
    rows: list[dict],
    *,
    limit: int = 0,
    per_class_limit: int = 0,
    minimum_score: float = 0.0,
) -> list[dict]:
    selected = []
    counts = defaultdict(int)
    for row in rows:
        if row["hard_score"] < minimum_score:
            continue
        label = row["predicted_label"]
        if per_class_limit and counts[label] >= per_class_limit:
            continue
        selected.append(row)
        counts[label] += 1
        if limit and len(selected) >= limit:
            break
    return selected


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    configured = load_mining_configuration(args.config)
    limit = configured["limit"] if args.limit is None else args.limit
    per_class_limit = (
        configured["per_class_limit"]
        if args.per_class_limit is None else args.per_class_limit
    )
    minimum_score = (
        configured["minimum_score"]
        if args.minimum_score is None else args.minimum_score
    )
    rare_class_quantile = (
        configured["rare_class_quantile"]
        if args.rare_class_quantile is None else args.rare_class_quantile
    )
    reason_threshold = (
        configured["reason_threshold"]
        if args.reason_threshold is None else args.reason_threshold
    )
    if limit < 0 or per_class_limit < 0:
        raise ValueError("candidate budgets must be non-negative")
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be in [0, 1]")
    if not 0 <= rare_class_quantile <= 1:
        raise ValueError("rare-class-quantile must be in [0, 1]")
    if not 0 <= reason_threshold <= 1:
        raise ValueError("reason-threshold must be in [0, 1]")
    if args.label_space_size is not None and args.label_space_size < 2:
        raise ValueError("label-space-size must be at least two")
    student_rows = (
        read_jsonl(args.student_jsonl)
        if args.student_jsonl
        else read_prediction_directory(args.prediction_dir)
    )
    teachers = {}
    teacher_inputs = []
    if args.teacher_jsonl:
        teacher_inputs.extend(read_jsonl(args.teacher_jsonl))
    if args.teacher_dir:
        teacher_inputs.extend(read_prediction_directory(args.teacher_dir))
    for row in teacher_inputs:
        value = row.get("result") or row
        sample_id = value.get("sample_id") or row.get("sample_id")
        if sample_id:
            teachers[str(sample_id)] = row
    ranked = mine(
        student_rows,
        teachers,
        rare_class_quantile=rare_class_quantile,
        label_space_size=args.label_space_size,
        before_teacher_weights=configured["before_teacher_weights"],
        after_teacher_weights=configured["after_teacher_weights"],
        reason_threshold=reason_threshold,
    )
    statistics = class_statistics(
        [row["predicted_label"] for row in ranked],
        rare_class_quantile,
    )
    statistics["label_space_sizes"] = sorted(
        {row["label_space_size"] for row in ranked}
    )
    if args.stats_output:
        destination = Path(args.stats_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    selected = select_budget(
        ranked,
        limit=limit,
        per_class_limit=per_class_limit,
        minimum_score=minimum_score,
    )
    write_jsonl(args.output, selected)
    enqueued = 0
    if args.review_db:
        store = ReviewStore(args.review_db)
        for row in selected:
            store.escalate_hard_sample(
                str(row["sample_id"]),
                hard_score=float(row["hard_score"]),
                priority=int(row["review_priority"]),
                reason=(
                    row["hard_reasons"][0]
                    if row["hard_reasons"] else "offline_hard_mining"
                ),
                payload={
                    "hard_components": row["hard_components"],
                    "predicted_label": row["predicted_label"],
                    "rare_class": row["rare_class"],
                    "rarity_score": row["rarity_score"],
                },
            )
            enqueued += 1
    log_event(
        "candidate_mining_complete",
        candidates=len(ranked),
        selected=len(selected),
        enqueued=enqueued,
        observed_classes=statistics["observed_class_count"],
        rare_labels=statistics["rare_labels"],
        output=args.output,
    )


if __name__ == "__main__":
    main()
