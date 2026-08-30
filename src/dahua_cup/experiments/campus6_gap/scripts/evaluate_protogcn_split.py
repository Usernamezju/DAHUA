#!/usr/bin/env python3
"""Single-GPU, unresampled ProtoGCN evaluation for a named annotation split."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    proto_root = Path(args.config).resolve().parents[2]
    sys.path.insert(0, str(proto_root))
    from protogcn.datasets import build_dataloader, build_dataset
    from protogcn.models import build_model

    cfg = Config.fromfile(args.config)
    cfg.data.test.ann_file = args.annotations
    cfg.data.test.split = args.split
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader = build_dataloader(
        dataset, videos_per_gpu=cfg.data.get("test_dataloader", {}).get("videos_per_gpu", 8),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 4), shuffle=False,
    )
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    device_id = device.index if device.index is not None else 0
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id]).eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.extend(model(return_loss=False, **batch))
    scores = np.asarray(scores)
    truth = np.asarray([item["label"] for item in dataset.video_infos], dtype=np.int64)
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((6, 6), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1
    report = {
        "split": args.split, "samples": int(len(truth)),
        "top1_accuracy": float((truth == predicted).mean()),
        "mean_class_accuracy": float(np.mean(np.diag(matrix) / np.maximum(matrix.sum(1), 1))),
        "confusion_matrix": matrix.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(scores, handle)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
