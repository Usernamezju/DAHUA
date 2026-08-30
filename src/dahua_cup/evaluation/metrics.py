"""Dependency-light classification, calibration and robustness metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


def _finite_probability(value, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return probability


def _distribution(value, labels: Sequence[str]) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw = {str(label): float(score) for label, score in value.items()}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = {
            str(item["label"]): float(
                item.get("probability", item.get("score", 0.0))
            )
            for item in value
        }
    else:
        raise ValueError("distribution must be a mapping or label/score list")
    unsupported = sorted(set(raw) - set(labels))
    if unsupported:
        raise ValueError(
            "distribution contains unsupported labels: "
            + ", ".join(unsupported)
        )
    if any(not math.isfinite(score) or score < 0 for score in raw.values()):
        raise ValueError("distribution contains invalid probabilities")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("distribution must contain positive probability mass")
    return {label: raw.get(label, 0.0) / total for label in labels}


def _prediction(row: Mapping, labels: Sequence[str]) -> tuple[str, float | None, dict | None]:
    full_distribution = _distribution(row.get("distribution"), labels)
    predicted = str(row.get("predicted_label", "")).strip()
    if not predicted and full_distribution:
        predicted = max(full_distribution, key=full_distribution.get)
    topk = row.get("topk") or row.get("top5") or []
    if not predicted and topk:
        predicted = str(topk[0].get("label", "")).strip()
    if predicted not in labels:
        raise ValueError(f"unsupported predicted label: {predicted}")
    if (
        full_distribution
        and predicted != max(full_distribution, key=full_distribution.get)
    ):
        raise ValueError(
            "predicted_label must match the full distribution argmax"
        )

    confidence = row.get("confidence")
    if confidence is None and full_distribution:
        confidence = full_distribution[predicted]
    if confidence is None and topk:
        confidence = topk[0].get("score", topk[0].get("probability"))
    confidence = (
        None
        if confidence is None
        else _finite_probability(confidence, "prediction confidence")
    )
    return predicted, confidence, full_distribution


def _classification_metrics(
    truths: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> dict:
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(truths, predictions):
        confusion[label_to_index[truth]][label_to_index[prediction]] += 1

    total = len(truths)
    correct = sum(confusion[index][index] for index in range(len(labels)))
    per_class = {}
    active = []
    for index, label in enumerate(labels):
        true_positive = confusion[index][index]
        false_negative = sum(confusion[index]) - true_positive
        false_positive = (
            sum(row[index] for row in confusion) - true_positive
        )
        true_negative = total - true_positive - false_negative - false_positive
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        false_positive_rate = (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
        }
        if support:
            active.append(per_class[label])

    macro_source = active or list(per_class.values())
    return {
        "sample_count": total,
        "label_count": len(labels),
        "observed_true_labels": len(active),
        "accuracy": correct / total if total else 0.0,
        "macro_precision": mean(item["precision"] for item in macro_source),
        "macro_recall": mean(item["recall"] for item in macro_source),
        "macro_f1": mean(item["f1"] for item in macro_source),
        "macro_false_positive_rate": mean(
            item["false_positive_rate"] for item in macro_source
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "rows": "true_label",
            "columns": "predicted_label",
            "labels": list(labels),
            "values": confusion,
        },
    }


def _calibration_metrics(
    correctness: Sequence[bool],
    confidences: Sequence[float],
    distributions: Sequence[dict | None],
    truths: Sequence[str],
    labels: Sequence[str],
    bins: int,
) -> dict:
    if bins < 2:
        raise ValueError("ECE bins must be at least two")
    calibration_bins = []
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row
            for row, confidence in enumerate(confidences)
            if confidence >= lower
            and (confidence < upper or (index == bins - 1 and confidence <= upper))
        ]
        if not members:
            calibration_bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        bin_confidence = mean(confidences[row] for row in members)
        bin_accuracy = mean(float(correctness[row]) for row in members)
        expected_calibration_error += (
            len(members) / len(confidences)
        ) * abs(bin_accuracy - bin_confidence)
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )

    brier_values = []
    for distribution, truth in zip(distributions, truths):
        if distribution is None:
            continue
        brier_values.append(
            sum(
                (
                    distribution[label] - float(label == truth)
                ) ** 2
                for label in labels
            )
        )
    return {
        "confidence_sample_count": len(confidences),
        "full_distribution_sample_count": len(brier_values),
        "ece_bins": bins,
        "ece": expected_calibration_error,
        "brier_score": mean(brier_values) if brier_values else None,
        "reliability_bins": calibration_bins,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_metrics(values: Sequence[float]) -> dict:
    if not values:
        return {"sample_count": 0}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency_ms must contain finite non-negative values")
    average = mean(values)
    return {
        "sample_count": len(values),
        "mean_ms": average,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "maximum_ms": max(values),
        "throughput_samples_per_second": 1000.0 / average if average else None,
    }


def _model_bundle(paths: Sequence[str | Path], maximum_bytes: int) -> dict:
    files = []
    total = 0
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"edge model file not found: {path}")
        size = path.stat().st_size
        files.append({"path": str(path), "size_bytes": size})
        total += size
    return {
        "files": files,
        "file_count": len(files),
        "total_size_bytes": total,
        "maximum_size_bytes": maximum_bytes,
        "verified": bool(files),
        "passed": total <= maximum_bytes if files else None,
    }


def evaluate_records(
    records: Sequence[Mapping],
    labels: Sequence[str],
    *,
    ece_bins: int = 15,
    slice_fields: Sequence[str] = ("occlusion", "viewpoint", "lighting"),
    edge_models: Sequence[str | Path] = (),
    maximum_edge_bytes: int = 50 * 1024 * 1024,
) -> dict:
    """Evaluate human-labelled predictions without sklearn dependencies."""
    labels = tuple(str(label) for label in labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be non-empty and unique")
    if not records:
        raise ValueError("evaluation requires at least one prediction record")

    sample_ids = set()
    truths = []
    predictions = []
    all_confidences = []
    confidence_correctness = []
    confidence_truths = []
    confidence_distributions = []
    latencies = []
    slices = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(records):
        sample_id = str(row.get("sample_id", f"row-{index}")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id}")
        sample_ids.add(sample_id)
        truth = str(row.get("true_label", "")).strip()
        if truth not in labels:
            raise ValueError(f"unsupported true label for {sample_id}: {truth}")
        prediction, confidence, distribution = _prediction(row, labels)
        row_index = len(truths)
        truths.append(truth)
        predictions.append(prediction)
        if confidence is not None:
            all_confidences.append(confidence)
            confidence_correctness.append(prediction == truth)
            confidence_truths.append(truth)
            confidence_distributions.append(distribution)
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        row_slices = dict(row.get("slices") or {})
        for field in slice_fields:
            value = row_slices.get(field, row.get(field))
            if value is not None and str(value).strip():
                slices[str(field)][str(value)].append(row_index)

    classification = _classification_metrics(truths, predictions, labels)
    calibration = (
        _calibration_metrics(
            confidence_correctness,
            all_confidences,
            confidence_distributions,
            confidence_truths,
            labels,
            ece_bins,
        )
        if all_confidences
        else {"confidence_sample_count": 0}
    )
    robustness = {}
    for field, groups in sorted(slices.items()):
        robustness[field] = {}
        for value, indices in sorted(groups.items()):
            metrics = _classification_metrics(
                [truths[index] for index in indices],
                [predictions[index] for index in indices],
                labels,
            )
            robustness[field][value] = {
                key: metrics[key]
                for key in ("sample_count", "accuracy", "macro_f1")
            }
    return {
        "schema_version": "competition_evaluation.v1",
        "labels": list(labels),
        "classification": classification,
        "calibration": calibration,
        "latency": _latency_metrics(latencies),
        "robustness_slices": robustness,
        "edge_model_bundle": _model_bundle(
            edge_models, maximum_edge_bytes
        ),
    }


def render_markdown_report(report: Mapping) -> str:
    classification = report["classification"]
    calibration = report["calibration"]
    latency = report["latency"]
    bundle = report["edge_model_bundle"]
    ece = calibration.get("ece")
    ece_text = f"{ece:.4f}" if ece is not None else "N/A"
    lines = [
        "# 校园行为语义理解算法测试报告",
        "",
        "## 总体指标",
        "",
        "| 样本数 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-FPR | ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {classification['sample_count']} "
            f"| {classification['accuracy']:.4f} "
            f"| {classification['macro_precision']:.4f} "
            f"| {classification['macro_recall']:.4f} "
            f"| {classification['macro_f1']:.4f} "
            f"| {classification['macro_false_positive_rate']:.4f} "
            f"| {ece_text} |"
        ),
        "",
        "## 每类指标",
        "",
        "| 类别 | Support | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in classification["per_class"].items():
        lines.append(
            f"| {label} | {values['support']} | {values['precision']:.4f} "
            f"| {values['recall']:.4f} | {values['f1']:.4f} "
            f"| {values['false_positive_rate']:.4f} |"
        )
    lines.extend(["", "## 混淆矩阵", ""])
    labels = classification["confusion_matrix"]["labels"]
    lines.append("| True \\ Pred | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for label, values in zip(
        labels, classification["confusion_matrix"]["values"]
    ):
        lines.append(
            f"| {label} | " + " | ".join(map(str, values)) + " |"
        )
    lines.extend(["", "## 鲁棒性分组", ""])
    if report["robustness_slices"]:
        lines.extend(
            [
                "| 分组字段 | 分组值 | 样本数 | Accuracy | Macro-F1 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for field, groups in report["robustness_slices"].items():
            for value, metrics in groups.items():
                lines.append(
                    f"| {field} | {value} | {metrics['sample_count']} "
                    f"| {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} |"
                )
    else:
        lines.append("未提供遮挡、视角或光照分组字段。")
    lines.extend(["", "## 时延与轻量化", ""])
    if latency.get("sample_count"):
        lines.append(
            f"平均时延 {latency['mean_ms']:.2f} ms，P95 "
            f"{latency['p95_ms']:.2f} ms，P99 {latency['p99_ms']:.2f} ms。"
        )
    else:
        lines.append("未提供 `latency_ms`，尚不能验收推理时延。")
    if bundle["verified"]:
        lines.append(
            f"边缘模型包共 {bundle['file_count']} 个文件，"
            f"{bundle['total_size_bytes'] / 1024 / 1024:.2f} MiB；"
            f"50 MiB 门禁：{'通过' if bundle['passed'] else '未通过'}。"
        )
    else:
        lines.append("未提供边缘模型文件，50 MiB 门禁尚未验证。")
    return "\n".join(lines) + "\n"
