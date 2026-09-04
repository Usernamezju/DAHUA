"""Evaluate an incremental ProtoGCN candidate and its regression baseline.

The command intentionally emits the small, stable metric contract consumed by
``release_gate.evaluate_release``.  It evaluates the candidate on the merged
pre-commit snapshot, then separately on the historical and newly reviewed
annotations; ``campus_all`` is not modified by this command.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from dahua_cup.semantic_teacher.schemas import LABELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--annotation", required=True, help="merged pre-commit annotation")
    parser.add_argument("--old-annotation", required=True)
    parser.add_argument("--new-annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lora", action="store_true")
    return parser


def _annotation_with_all(source: Path, destination: Path) -> Path:
    with source.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("annotations"), list):
        raise ValueError(f"invalid Campus6 annotation: {source}")
    split = dict(value.get("split") or {})
    if not all(name in split for name in ("train", "val", "test")):
        raise ValueError(f"Campus6 annotation is missing train/val/test: {source}")
    split["all"] = list(split["train"]) + list(split["val"]) + list(split["test"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump({"annotations": value["annotations"], "split": split}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def _model(config: Path, checkpoint: Path, device: str, *, lora: bool):
    import mmcv
    from mmcv.runner import load_checkpoint
    from protogcn.models import build_recognizer

    cfg = mmcv.Config.fromfile(str(config))
    model = build_recognizer(cfg.model)
    if lora:
        from dahua_cup.semantic_teacher.incremental.lora import install_classifier_lora

        install_classifier_lora(
            model,
            rank=int(os.environ.get("DAHUA_PROTOGCN_LORA_RANK", "8")),
            alpha=float(os.environ.get("DAHUA_PROTOGCN_LORA_ALPHA", "16")),
        )
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=False)
    model.to(device)
    model.eval()
    return cfg, model


def _metric(labels: list[int], predictions: list[int], probabilities: list[np.ndarray]) -> dict:
    if not labels:
        raise ValueError("evaluation split is empty")
    class_count = len(LABELS)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, predicted in zip(labels, predictions):
        if 0 <= truth < class_count and 0 <= predicted < class_count:
            confusion[truth, predicted] += 1
    precision, recall, f1 = [], [], []
    for index in range(class_count):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = predicted == truth
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 16)[:-1], np.linspace(0.0, 1.0, 16)[1:]):
        selected = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if selected.any():
            ece += float(selected.mean()) * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
    return {
        "sample_count": len(labels),
        "accuracy": float(correct.mean()),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "dangerous_recall": float(np.mean(recall[4:6])),
        "ece": float(ece),
        "per_class": {
            LABELS[index]: {"precision": precision[index], "recall": recall[index], "f1": f1[index]}
            for index in range(class_count)
        },
    }


def _evaluate(model, cfg, annotation: Path, *, device: str, batch_size: int, workers: int) -> dict:
    import torch
    from dahua_cup.semantic_teacher.distillation.campus6_compression import _import_protogcn, _loader, _unwrap, evaluation_scores

    _import_protogcn(Path.cwd())
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for batch in _loader(cfg, "all", annotation, batch_size, workers):
            keypoint, label = _unwrap(batch, device)
            scores = evaluation_scores(model, keypoint)
            probability = torch.softmax(scores, dim=1)
            labels.extend(label.detach().cpu().tolist())
            predictions.extend(probability.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(probability.detach().cpu().numpy())
    return _metric(labels, predictions, probabilities)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    candidate = Path(args.candidate_checkpoint).expanduser().resolve()
    baseline = Path(args.baseline_checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    for path in (candidate, baseline, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    work = Path(args.output).expanduser().resolve().parent / "evaluation_inputs"
    merged = _annotation_with_all(Path(args.annotation).expanduser().resolve(), work / "merged.pkl")
    old = _annotation_with_all(Path(args.old_annotation).expanduser().resolve(), work / "old.pkl")
    new = _annotation_with_all(Path(args.new_annotation).expanduser().resolve(), work / "new.pkl")
    # Both checkpoints use the same graph.  LoRA loading is enabled for the
    # candidate and also for the baseline so legacy classifier keys are mapped
    # through the zero-initialised adapter without changing baseline logits.
    from dahua_cup.semantic_teacher.distillation.campus6_compression import _import_protogcn

    _import_protogcn(Path.cwd())
    candidate_cfg, candidate_model = _model(config, candidate, args.device, lora=args.lora)
    baseline_cfg, baseline_model = _model(config, baseline, args.device, lora=args.lora)
    candidate_global = _evaluate(candidate_model, candidate_cfg, merged, device=args.device, batch_size=args.batch_size, workers=args.workers)
    candidate_old = _evaluate(candidate_model, candidate_cfg, old, device=args.device, batch_size=args.batch_size, workers=args.workers)
    candidate_new = _evaluate(candidate_model, candidate_cfg, new, device=args.device, batch_size=args.batch_size, workers=args.workers)
    baseline_global = _evaluate(baseline_model, baseline_cfg, merged, device=args.device, batch_size=args.batch_size, workers=args.workers)
    baseline_old = _evaluate(baseline_model, baseline_cfg, old, device=args.device, batch_size=args.batch_size, workers=args.workers)
    baseline_new = _evaluate(baseline_model, baseline_cfg, new, device=args.device, batch_size=args.batch_size, workers=args.workers)
    result = {
        "schema_version": "incremental_evaluation.v1",
        "candidate": {
            "global_macro_f1": candidate_global["macro_f1"],
            "new_macro_f1": candidate_new["macro_f1"],
            "old_macro_f1": candidate_old["macro_f1"],
            "dangerous_recall": candidate_global["dangerous_recall"],
            "ece": candidate_global["ece"],
            "edge_size_bytes": candidate.stat().st_size,
        },
        "baseline": {
            "global_macro_f1": baseline_global["macro_f1"],
            "new_macro_f1": baseline_new["macro_f1"],
            "old_macro_f1": baseline_old["macro_f1"],
            "dangerous_recall": baseline_global["dangerous_recall"],
            "ece": baseline_global["ece"],
        },
        "splits": {
            "candidate": {"global": candidate_global, "old": candidate_old, "new": candidate_new},
            "baseline": {"global": baseline_global, "old": baseline_old, "new": baseline_new},
        },
        "candidate_checkpoint": str(candidate),
        "baseline_checkpoint": str(baseline),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
