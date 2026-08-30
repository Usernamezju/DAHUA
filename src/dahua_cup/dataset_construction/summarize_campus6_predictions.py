#!/usr/bin/env python3
"""Write reproducible Campus6 accuracy/F1 metrics from a ProtoGCN prediction pkl."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


LABELS = ("normal_walk", "normal_run", "playful_chase", "playful_push", "conflict_chase", "conflict_push")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.annotations.open("rb") as stream:
        annotations = pickle.load(stream)
    ids = set(annotations["split"][args.split])
    expected = [item for item in annotations["annotations"] if item["frame_dir"] in ids]
    with args.predictions.open("rb") as stream:
        scores = pickle.load(stream)
    scores = np.asarray(scores)
    if len(expected) != len(scores):
        raise RuntimeError(f"prediction/annotation count mismatch: {len(scores)} != {len(expected)}")
    truth = np.asarray([item["label"] for item in expected], dtype=np.int64)
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1
    per_class = {}
    f1s = []
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class[label] = {"support": int(matrix[index].sum()), "correct": tp,
                            "accuracy": recall, "precision": precision, "recall": recall, "f1": f1}
        if matrix[index].sum():
            f1s.append(f1)
    report = {"split": args.split, "samples": int(len(truth)),
              "top1_accuracy": float((truth == predicted).mean()),
              "macro_f1": float(np.mean(f1s)), "per_class": per_class,
              "confusion_matrix": matrix.tolist(), "labels": list(LABELS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
