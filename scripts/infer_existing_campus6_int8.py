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
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    with Path(args.annotations).open("rb") as stream:
        payload = pickle.load(stream)
    annotations = {
        str(item["frame_dir"]): item for item in payload["annotations"]
    }
    order = [str(name) for name in payload["split"]["all"]]
    if len(order) != len(annotations):
        raise ValueError("Campus6 all split must contain every annotation once")

    model, checkpoint_format, metadata = initialize_model(
        Path(args.config), Path(args.checkpoint), args.device, "quantized"
    )
    import mmcv
    import torch

    from dahua_cup.semantic_teacher.distillation.campus6_compression import (
        _loader,
        _unwrap,
        evaluation_scores,
    )

    # Use the exact deterministic loader and multi-clip aggregation used to
    # produce M1FKD.metrics.json.  Calling inference_recognizer per sample is
    # not equivalent to this acceptance protocol.
    cfg = mmcv.Config.fromfile(str(Path(args.config).resolve()))
    score_batches, label_batches = [], []
    processed = 0
    model.eval()
    with torch.no_grad():
        loader = _loader(
            cfg, "all", Path(args.annotations).resolve(),
            args.batch_size, args.workers,
        )
        for batch in loader:
            keypoint, label = _unwrap(batch, args.device)
            scores = evaluation_scores(model, keypoint)
            score_batches.append(scores.detach().cpu().numpy().astype(np.float32))
            label_batches.append(label.detach().cpu().numpy().astype(np.int64))
            processed += int(label.numel())
            print(f"processed {processed}/{len(order)}", flush=True)
    probabilities = np.concatenate(score_batches, axis=0)
    ground_truth = np.concatenate(label_batches, axis=0)
    if probabilities.shape != (len(order), len(LABELS)):
        raise ValueError(
            f"evaluation produced {probabilities.shape}, expected "
            f"({len(order)}, {len(LABELS)})"
        )

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
        "order": "annotations.split.all",
        "evaluation_protocol": "campus6_compression._loader+evaluation_scores",
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
