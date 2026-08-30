#!/usr/bin/env python3
"""Generate missing per-sample M1FKD INT8 probabilities from existing skeletons."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.rtmpose17_student_worker import initialize_model
from dahua_cup.semantic_teacher.schemas import LABELS


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", required=True)
    value.add_argument("--config", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument(
        "--calibration-split", default="val",
        choices=("train", "val", "test", "all"),
        help="split used only to fit one confidence-temperature scalar",
    )
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    with Path(args.annotations).open("rb") as stream:
        payload = pickle.load(stream)
    annotations = {
        str(item["frame_dir"]): item for item in payload["annotations"]
    }
    # MMAction PoseDataset retains annotation-list order after applying a
    # split membership filter; it does not reorder rows to split["all"].
    order = [str(item["frame_dir"]) for item in payload["annotations"]]
    split_all = [str(name) for name in payload["split"]["all"]]
    if len(order) != len(annotations) or set(order) != set(split_all):
        raise ValueError("Campus6 all split must contain every annotation once")

    model, checkpoint_format, metadata = initialize_model(
        Path(args.config), Path(args.checkpoint), args.device, "quantized"
    )
    import mmcv
    import torch

    from dahua_cup.semantic_teacher.distillation.campus6_compression import (
        _loader,
        _unwrap,
        raw_logits,
    )

    # Use the exact deterministic loader and multi-clip aggregation used to
    # produce M1FKD.metrics.json.  Calling inference_recognizer per sample is
    # not equivalent to this acceptance protocol.
    cfg = mmcv.Config.fromfile(str(Path(args.config).resolve()))
    logit_batches, label_batches = [], []
    processed = 0
    model.eval()
    with torch.no_grad():
        loader = _loader(
            cfg, "all", Path(args.annotations).resolve(),
            args.batch_size, args.workers,
        )
        for batch in loader:
            keypoint, label = _unwrap(batch, args.device)
            logits = raw_logits(model, keypoint)
            logit_batches.append(logits.detach().cpu().numpy().astype(np.float64))
            label_batches.append(label.detach().cpu().numpy().astype(np.int64))
            processed += int(label.numel())
            print(f"processed {processed}/{len(order)}", flush=True)
    logits = np.concatenate(logit_batches, axis=0)
    ground_truth = np.concatenate(label_batches, axis=0)
    if logits.shape != (len(order), len(LABELS)):
        raise ValueError(
            f"evaluation produced {logits.shape}, expected "
            f"({len(order)}, {len(LABELS)})"
        )

    calibration_names = set(
        str(name) for name in payload["split"][args.calibration_split]
    )
    calibration_rows = np.asarray(
        [name in calibration_names for name in order], dtype=bool
    )
    if not calibration_rows.any():
        raise ValueError("confidence calibration split is empty")

    def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
        scaled = values / float(temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        exponent = np.exp(scaled)
        return exponent / exponent.sum(axis=1, keepdims=True)

    # Temperature scaling changes confidence only; argmax labels remain the
    # direct output of the existing INT8 model.  A logarithmic search is
    # dependency-free and deterministic for this one scalar.
    candidates = np.geomspace(0.05, 100000.0, num=4096)
    calibration_logits = logits[calibration_rows]
    calibration_labels = ground_truth[calibration_rows]
    losses = []
    for temperature in candidates:
        values = softmax(calibration_logits, float(temperature))
        selected = values[np.arange(len(values)), calibration_labels]
        losses.append(float(-np.log(np.maximum(selected, 1e-300)).mean()))
    temperature = float(candidates[int(np.argmin(losses))])
    probabilities = softmax(logits, temperature).astype(np.float32)
    if not np.array_equal(logits.argmax(axis=1), probabilities.argmax(axis=1)):
        raise RuntimeError("temperature calibration changed an INT8 Top-1 label")

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(probabilities, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    predicted = probabilities.argmax(axis=1)
    summary = {
        "schema_version": "campus6_m1fkd_predictions.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_format": checkpoint_format,
        "deployment_metadata": metadata,
        "annotations": str(Path(args.annotations).resolve()),
        "order": "annotations",
        "evaluation_protocol": "campus6_compression._loader+raw_logits",
        "confidence_calibration": {
            "method": "temperature_scaling",
            "split": args.calibration_split,
            "sample_count": int(calibration_rows.sum()),
            "temperature": temperature,
            "validation_nll": min(losses),
            "top1_preserved": True,
        },
        "shape": list(probabilities.shape),
        "correct": int((predicted == ground_truth).sum()),
        "total": len(order),
        "accuracy": float((predicted == ground_truth).mean()),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
