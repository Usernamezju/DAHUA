#!/usr/bin/env python3
"""Evaluate every INT8 model, each pair, and the three-model ensemble.

Ensembling is deliberately at output level: independent quantized ONNX models
produce six-class scores, then calibrated probabilities are averaged.  Raw
INT8 tensors from different models must never be averaged.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(scores: np.ndarray, labels: np.ndarray, classes: int) -> dict:
    prediction = scores.argmax(1)
    recalls, f1s = [], []
    for index in range(classes):
        truth, chosen = labels == index, prediction == index
        tp = int((truth & chosen).sum())
        recall, precision = tp / max(int(truth.sum()), 1), tp / max(int(chosen.sum()), 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "top1": float((prediction == labels).mean()), "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)), "per_class_recall": recalls,
    }


def parse_model(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("model must be NAME=MODEL.onnx")
    name, path = text.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=parse_model,
                        help="repeat NAME=quantized_model.onnx")
    parser.add_argument("--evaluation-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--score-kind", choices=["logits", "probabilities"], default="logits")
    args = parser.parse_args()
    if len(args.model) not in (2, 3):
        raise SystemExit("provide exactly two or three INT8 models")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Install onnxruntime in the evaluation environment.") from exc
    archive = np.load(args.evaluation_npz, allow_pickle=False)
    data, labels = np.asarray(archive["data"], dtype=np.float32), np.asarray(archive["label"], dtype=np.int64)
    outputs: dict[str, np.ndarray] = {}
    for name, model_path in args.model:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = args.input_name or session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        scores = np.concatenate([session.run([output_name], {input_name: item[None]})[0] for item in data])
        if scores.shape != (len(data), args.num_classes):
            raise SystemExit(f"{name} produced {scores.shape}; expected ({len(data)}, {args.num_classes})")
        outputs[name] = scores if args.score_kind == "probabilities" else softmax(scores)
    results = []
    names = list(outputs)
    for count in range(1, len(names) + 1):
        for subset in itertools.combinations(names, count):
            fused = np.mean([outputs[name] for name in subset], axis=0)
            row = {"ensemble": "+".join(subset), "members": len(subset), **evaluate(fused, labels, args.num_classes)}
            results.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "int8_ensemble_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "int8_ensemble_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ensemble", "members", "top1", "macro_recall", "macro_f1", "per_class_recall"])
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
