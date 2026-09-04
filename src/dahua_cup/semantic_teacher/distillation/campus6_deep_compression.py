"""Run paper-faithful Deep Compression on the verified Campus6 M0 model.

Pipeline: global magnitude pruning with persistent masks, masked recovery,
layer-wise k-means weight sharing with trainable centroids, then offline
Huffman packing of codebook assignments and sparse location deltas.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .campus6_compression import (
    _all_annotation,
    _build_model,
    _import_protogcn,
    _load_config,
    _train_loader,
    _unwrap,
    evaluate_splits,
    evaluate_split,
    raw_logits,
)
from .deep_compression import (
    apply_global_magnitude_pruning,
    apply_trained_weight_sharing,
    campus6_training_only_state_keys,
    export_huffman_package,
    export_huffman_runtime_residual,
    materialize_masks,
    materialize_weight_sharing,
)
from .logits_kd import LogitsKDConfig, freeze_teacher, logits_kd_loss


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True)
    value.add_argument("--checkpoint", required=True, help="verified M0 FP32 checkpoint")
    value.add_argument("--ann-file", required=True)
    value.add_argument("--work-dir", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--seed", type=int, default=20260903)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--sparsity", type=float, default=0.85)
    value.add_argument("--max-layer-sparsity", type=float, default=0.90)
    value.add_argument("--prune-epochs", type=int, default=12)
    value.add_argument("--sharing-epochs", type=int, default=16)
    value.add_argument("--prune-lr", type=float, default=1e-4)
    value.add_argument("--sharing-lr", type=float, default=5e-5)
    value.add_argument("--conv-bits", type=int, choices=range(2, 9), default=8)
    value.add_argument("--head-bits", type=int, choices=range(2, 9), default=8)
    value.add_argument("--kmeans-iterations", type=int, default=20)
    value.add_argument("--temperature", type=float, default=4.0)
    value.add_argument("--kd-weight", type=float, default=1.0)
    value.add_argument("--ce-weight", type=float, default=1.0)
    value.add_argument("--dry-run", action="store_true")
    return value


def _seed(value: int) -> None:
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _train_epoch(model, teacher, loader, optimizer, kd: LogitsKDConfig, device: str) -> dict:
    import torch

    model.train()
    freeze_teacher(teacher)
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_kd": 0.0, "samples": 0}
    for batch in loader:
        keypoint, label = _unwrap(batch, device)
        optimizer.zero_grad(set_to_none=True)
        student = raw_logits(model, keypoint)
        with torch.no_grad():
            teacher_logits = raw_logits(teacher, keypoint)
        pieces = logits_kd_loss(student, teacher_logits, label, kd)
        pieces["loss"].backward()
        optimizer.step()
        count = int(label.numel())
        totals["samples"] += count
        for key in ("loss", "loss_ce", "loss_kd"):
            totals[key] += float(pieces[key].detach().item()) * count
    return {key: (value / totals["samples"] if key != "samples" else value) for key, value in totals.items()}


def _recover(model, teacher, loader, cfg, ann_file, epochs: int, lr: float, kd, args, phase: str) -> tuple[list[dict], dict]:
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    history, best_state, best = [], None, None
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch, **_train_epoch(model, teacher, loader, optimizer, kd, args.device)}
        row["val"] = evaluate_split(model, cfg, "val", ann_file, device=args.device, batch_size=args.batch_size, workers=args.workers)
        history.append(row)
        if best is None or row["val"]["accuracy"] > best["val"]["accuracy"]:
            best_state = copy.deepcopy(model.state_dict())
            best = {"phase": phase, "epoch": epoch, "val": row["val"]}
    if best_state is None:
        raise RuntimeError(f"{phase} did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    return history, best


def main(argv=None) -> None:
    import torch

    args = parser().parse_args(argv)
    if args.prune_epochs < 1 or args.sharing_epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid training settings")
    _seed(args.seed)
    root = Path(__file__).resolve().parents[3]
    _import_protogcn(root)
    config, checkpoint, ann_file = _load_config(Path(args.config)), Path(args.checkpoint).resolve(), Path(args.ann_file).resolve()
    if not checkpoint.is_file() or not ann_file.is_file():
        raise FileNotFoundError("checkpoint and annotation file must exist")
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    all_ann = _all_annotation(ann_file, work_dir / "annotations_with_all.pkl")
    contract = {
        "paper": "Han et al., Deep Compression, ICLR 2016",
        "baseline_checkpoint": str(checkpoint),
        "ann_file": str(ann_file),
        "all_ann_file": str(all_ann),
        "seed": args.seed,
        "splits": ["train", "val", "test", "all"],
        "target_sparsity": args.sparsity,
        "max_layer_sparsity": args.max_layer_sparsity,
        "weight_sharing": {"conv_bits": args.conv_bits, "head_bits": args.head_bits, "kmeans_initialization": "linear", "kmeans_iterations": args.kmeans_iterations},
        "recovery": {"prune_epochs": args.prune_epochs, "sharing_epochs": args.sharing_epochs, "prune_lr": args.prune_lr, "sharing_lr": args.sharing_lr, "distillation": asdict(LogitsKDConfig(args.temperature, args.ce_weight, args.kd_weight))},
        "huffman": "codebook indices and flattened sparse-position deltas",
        "scope": "backbone plus cls_head.fc_cls; GAP semantic and CSC auxiliary branches excluded from deployment compression",
    }
    (work_dir / "experiment_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **contract}, ensure_ascii=False))
        return
    baseline = _build_model(config, checkpoint, args.device)
    teacher = copy.deepcopy(baseline)
    freeze_teacher(teacher)
    model = copy.deepcopy(baseline)
    train_loader = _train_loader(config, ann_file, args.batch_size, args.workers, args.seed)
    kd = LogitsKDConfig(args.temperature, args.ce_weight, args.kd_weight)

    masks, pruning = apply_global_magnitude_pruning(model, args.sparsity, args.max_layer_sparsity)
    (work_dir / "pruning_report.json").write_text(json.dumps(pruning.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prune_history, prune_best = _recover(model, teacher, train_loader, config, all_ann, args.prune_epochs, args.prune_lr, kd, args, "masked_pruning_recovery")
    materialize_masks(model)

    sharing = apply_trained_weight_sharing(model, masks, args.conv_bits, args.head_bits, args.kmeans_iterations)
    sharing_manifest = {name: {key: value for key, value in row.items() if key != "parameterization"} for name, row in sharing.items()}
    (work_dir / "weight_sharing_init.json").write_text(json.dumps(sharing_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sharing_history, sharing_best = _recover(model, teacher, train_loader, config, all_ann, args.sharing_epochs, args.sharing_lr, kd, args, "trained_weight_sharing")
    package = export_huffman_package(model, masks, sharing, work_dir / "M0_deep_compression.huffman.bin")
    materialize_weight_sharing(model)
    residual = export_huffman_runtime_residual(
        model,
        work_dir / "M0_deep_compression.huffman.bin",
        work_dir / "M0_deep_compression.residual.pth",
        excluded_state_keys=campus6_training_only_state_keys(model),
    )
    metrics = evaluate_splits(model, config, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers)
    final_checkpoint = work_dir / "M0_deep_compression.materialized.pth"
    torch.save({"state_dict": model.state_dict(), "meta": {"contract": contract, "pruning": pruning.as_dict(), "prune_best": prune_best, "sharing_best": sharing_best, "huffman_package": package}}, final_checkpoint)
    summary = {
        "status": "completed",
        "model": "M0_DeepCompression",
        "checkpoint": str(final_checkpoint),
        "checkpoint_size_mib": final_checkpoint.stat().st_size / 1024.0 / 1024.0,
        "metrics": metrics,
        "pruning": pruning.as_dict(),
        "prune_recovery": prune_history,
        "sharing_recovery": sharing_history,
        "best_validation": {"prune": prune_best, "sharing": sharing_best},
        "huffman_package": package,
        "huffman_residual": residual,
        "contract": contract,
        "caveat": "The materialized .pth restores ordinary dense weights for PyTorch validation. The .huffman.bin package is the paper-style deployment-storage artifact and requires sparse/codebook-aware inference to realize speedup.",
    }
    (work_dir / "DeepCompression.metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "metrics": metrics, "pruning": pruning.as_dict(), "huffman_package": package}, ensure_ascii=False))


if __name__ == "__main__":
    main()
