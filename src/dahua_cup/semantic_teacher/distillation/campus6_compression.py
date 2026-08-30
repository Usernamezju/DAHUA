"""Run comparable Campus6 QAT, structured-pruning and logits-KD experiments.

M0 is evaluated read-only. M1 applies QAT to M0; M1KD is the same M1
structure trained with QAT plus frozen-M0 raw-logits distillation. M2
physically removes structured channels from a fresh M0 copy, recovers it, then
applies QAT. M3 starts from the same recovered M2 structure and jointly
applies fake-INT8 QAT and logits KD from frozen M0. Every stage is evaluated
with the same non-augmented train/val/test/all protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .logits_kd import (
    FeatureKDConfig,
    LogitsKDConfig,
    ThreeStageFeatureAligner,
    WeightActivationFakeQuant,
    checkpoint_size_mb,
    export_quantized_state_dict,
    export_int8_state_dict,
    freeze_teacher,
    load_int8_state_dict,
    logits_kd_loss,
)
from .structured_pruning import StructuredPruningConfig, build_structurally_pruned_protogcn


SPLITS = ("train", "val", "test", "all")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True)
    value.add_argument("--checkpoint", required=True, help="M0 FP32 checkpoint")
    value.add_argument("--ann-file", required=True)
    value.add_argument("--work-dir", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--seed", type=int, default=20260829)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--qat-epochs", type=int, default=12)
    value.add_argument("--recovery-epochs", type=int, default=8)
    value.add_argument("--qat-lr", type=float, default=3e-4)
    value.add_argument("--recovery-lr", type=float, default=3e-4)
    value.add_argument("--grad-clip-norm", type=float, default=5.0)
    value.add_argument("--prune-ratio", type=float, default=0.40)
    value.add_argument("--temperature", type=float, default=4.0)
    value.add_argument("--kd-weight", type=float, default=1.0)
    value.add_argument("--feature-kd-weight", type=float, default=0.1)
    value.add_argument("--quant-bits", type=int, choices=(4, 8), default=8)
    value.add_argument("--ce-weight", type=float, default=1.0)
    value.add_argument("--stages", nargs="+", choices=("M0", "M1", "M1KD", "M1FKD", "M1I4", "M1I4KD", "M1I4FKD", "M2", "M3"), default=("M0", "M1", "M2", "M3"))
    value.add_argument("--dry-run", action="store_true")
    return value


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _import_protogcn(root: Path) -> None:
    import sys

    path = str(root / "gcn_models" / "ProtoGCN")
    if path not in sys.path:
        sys.path.insert(0, path)
    # Register the Campus6 GAP recognizer before building the configured model.
    from protogcn.models.recognizers.recognizergcn_gap import RecognizerGCNGAP  # noqa: F401


def _load_config(path: Path):
    import mmcv

    return mmcv.Config.fromfile(str(path))


def _build_model(cfg, checkpoint: Path, device: str):
    from mmcv.runner import load_checkpoint
    from protogcn.models import build_recognizer

    model = build_recognizer(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.to(device)
    return model


def _eval_pipeline(cfg):
    return copy.deepcopy(cfg.get("eval_pipeline") or cfg.data.test.pipeline)


def _all_annotation(ann_file: Path, destination: Path) -> Path:
    import mmcv

    value = mmcv.load(str(ann_file))
    if not isinstance(value, dict) or "split" not in value or "annotations" not in value:
        raise ValueError("Campus6 annotation must contain split and annotations")
    split = dict(value["split"])
    split["all"] = list(split["train"]) + list(split["val"]) + list(split["test"])
    if len(split["all"]) != len(set(split["all"])):
        raise ValueError("ALL split contains duplicate sample identifiers")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump({"split": split, "annotations": value["annotations"]}, str(destination))
    return destination


def _dataset_cfg(cfg, split: str, ann_file: Path):
    item = copy.deepcopy(cfg.data.test)
    item["ann_file"] = str(ann_file)
    item["split"] = split
    item["pipeline"] = _eval_pipeline(cfg)
    item.pop("class_prob", None)
    return item


def _loader(cfg, split: str, ann_file: Path, batch_size: int, workers: int):
    from protogcn.datasets import build_dataset
    from protogcn.datasets.builder import build_dataloader

    dataset = build_dataset(_dataset_cfg(cfg, split, ann_file), dict(test_mode=True))
    return build_dataloader(
        dataset, videos_per_gpu=batch_size, workers_per_gpu=workers,
        shuffle=False, drop_last=False, pin_memory=True,
    )


def _train_loader(cfg, ann_file: Path, batch_size: int, workers: int, seed: int):
    from protogcn.datasets import build_dataset
    from protogcn.datasets.builder import build_dataloader

    item = copy.deepcopy(cfg.data.train)
    item["ann_file"] = str(ann_file)
    # Keep every compression stage on exactly the declared 252-sample train
    # split.  The original Campus6 recipe repeats a rare class; that changes
    # the number and empirical distribution of examples per epoch, so it is
    # intentionally disabled for this controlled M0--M3 comparison.
    item.pop("class_prob", None)
    dataset = build_dataset(item, dict(test_mode=False))
    return build_dataloader(
        dataset, videos_per_gpu=batch_size, workers_per_gpu=workers,
        shuffle=True, seed=seed, drop_last=False, pin_memory=True,
    )


def _unwrap(batch, device: str):
    # This project's MMCV collate returns a stacked Tensor in ``.data``.  Do
    # not index it: ``.data[0]`` silently discards the other B-1 samples.
    keypoint = batch["keypoint"].data.to(device, non_blocking=True)
    # With MMCV's non-stacked DataContainer, a single sample arrives as
    # [clips, persons, time, joints, channels] rather than having an explicit
    # leading batch axis.  Preserve a true batched tensor for the recognizer.
    if keypoint.ndim == 5:
        keypoint = keypoint.unsqueeze(0)
    if keypoint.ndim != 6:
        raise ValueError(
            "Expected keypoints shaped [batch, clips, persons, time, joints, channels], got "
            f"{tuple(keypoint.shape)}"
        )
    label = batch["label"].data.to(device, non_blocking=True).reshape(-1).long()
    if keypoint.shape[0] != label.numel():
        raise ValueError(
            f"Batch/label mismatch: {keypoint.shape[0]} samples but {label.numel()} labels"
        )
    return keypoint, label


def raw_logits(model, keypoint):
    """Return raw per-clip classifier logits for CE/KD training.

    The training pipeline deliberately contains one clip.  Keeping this
    contract avoids distilling probabilities that were already aggregated by
    the evaluation-time multi-clip policy.
    """
    if keypoint.shape[1] != 1:
        raise ValueError("CE/KD training expects exactly one training clip")
    feature, _graph = model.extract_feat(keypoint[:, 0])
    return model.cls_head(feature)


def evaluation_scores(model, keypoint):
    """Apply the model's standard multi-clip evaluation aggregation.

    Test pipelines may emit several deterministic clips for each sample.  We
    keep their logits separate until ``average_clip`` so M0's reported metric
    is identical to ordinary ProtoGCN evaluation (normally mean probability),
    while training and logits distillation retain true pre-softmax logits.
    """
    batch_size, clip_count = keypoint.shape[:2]
    flattened = keypoint.reshape((batch_size * clip_count,) + tuple(keypoint.shape[2:]))
    feature, _graph = model.extract_feat(flattened)
    logits = model.cls_head(feature).reshape(batch_size, clip_count, -1)
    return model.average_clip(logits)


def evaluate_split(model, cfg, split: str, ann_file: Path, *, device: str, batch_size: int, workers: int) -> dict:
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in _loader(cfg, split, ann_file, batch_size, workers):
            keypoint, label = _unwrap(batch, device)
            predicted = evaluation_scores(model, keypoint).argmax(dim=1)
            correct += int((predicted == label).sum().item())
            total += int(label.numel())
    if total == 0:
        raise ValueError(f"{split} evaluation split is empty")
    return {"correct": correct, "total": total, "accuracy": correct / total}


def evaluate_splits(model, cfg, ann_file: Path, *, device: str, batch_size: int, workers: int) -> dict:
    result = {
        split: evaluate_split(
            model, cfg, split, ann_file, device=device,
            batch_size=batch_size, workers=workers,
        )
        for split in SPLITS
    }
    expected = result["train"]["total"] + result["val"]["total"] + result["test"]["total"]
    if result["all"]["total"] != expected:
        raise AssertionError(
            "ALL must contain exactly train + val + test samples: "
            f"all={result['all']['total']}, expected={expected}, details={result}"
        )
    return result


def _supervised_epoch(model, loader, optimizer, device: str, *, teacher=None, kd=None, feature_aligner=None, grad_clip_norm=None) -> dict:
    import torch
    import torch.nn.functional as functional

    model.train()
    if teacher is not None:
        freeze_teacher(teacher)
    values = {"loss": 0.0, "loss_ce": 0.0, "loss_kd": 0.0, "loss_feature": 0.0, "samples": 0}
    for batch in loader:
        keypoint, label = _unwrap(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if feature_aligner is not None:
            feature_aligner.clear()
        student = raw_logits(model, keypoint)
        if teacher is None:
            loss = functional.cross_entropy(student, label)
            parts = {"loss": loss, "loss_ce": loss, "loss_kd": loss.detach() * 0.0}
        else:
            with torch.no_grad():
                teacher_logits = raw_logits(teacher, keypoint)
            parts = logits_kd_loss(student, teacher_logits, label, kd)
            loss = parts["loss"]
        if feature_aligner is not None:
            feature_loss = feature_aligner.loss()
            loss = loss + feature_aligner.config.weight * feature_loss
            parts["loss"] = loss
            parts["loss_feature"] = feature_loss
        else:
            parts["loss_feature"] = loss.detach() * 0.0
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss; lower the QAT learning rate")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        count = int(label.numel())
        values["samples"] += count
        for name in ("loss", "loss_ce", "loss_kd", "loss_feature"):
            values[name] += float(parts[name].detach().item()) * count
    if not values["samples"]:
        raise ValueError("training loader is empty")
    return {name: (value / values["samples"] if name != "samples" else value) for name, value in values.items()}


def _train(model, loader, epochs: int, lr: float, device: str, *, teacher=None, kd=None, feature_aligner=None, grad_clip_norm=None) -> list[dict]:
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    return [
        {"epoch": epoch + 1, **_supervised_epoch(model, loader, optimizer, device, teacher=teacher, kd=kd, feature_aligner=feature_aligner, grad_clip_norm=grad_clip_norm)}
        for epoch in range(epochs)
    ]


def _train_selecting_best_val(model, loader, cfg, ann_file: Path, epochs: int, lr: float, device: str, *, batch_size: int, workers: int, teacher=None, kd=None, feature_aligner=None, grad_clip_norm=None) -> tuple[list[dict], dict]:
    """Train and restore the epoch with best fixed-split validation accuracy.

    Validation is used only for model selection.  Test and ALL remain unseen
    until one final report is generated from the restored best checkpoint.
    """
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    history, best_state, best = [], None, None
    for epoch in range(epochs):
        row = {"epoch": epoch + 1, **_supervised_epoch(model, loader, optimizer, device, teacher=teacher, kd=kd, feature_aligner=feature_aligner, grad_clip_norm=grad_clip_norm)}
        row["val"] = evaluate_split(model, cfg, "val", ann_file, device=device, batch_size=batch_size, workers=workers)
        history.append(row)
        if best is None or row["val"]["accuracy"] > best["val"]["accuracy"]:
            best_state = copy.deepcopy(model.state_dict())
            best = {"epoch": epoch + 1, "val": row["val"]}
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    return history, best


def _save_stage(work_dir: Path, name: str, model, metrics: dict, metadata: dict, *, quant_bits: int | None = None) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    if quant_bits:
        checkpoint = work_dir / f"{name}.int{quant_bits}.pt"
        export_quantized_state_dict(model, checkpoint, metadata, num_bits=quant_bits)
    else:
        checkpoint = work_dir / f"{name}.fp32.pth"
        import torch
        torch.save({"state_dict": model.state_dict(), "meta": metadata}, checkpoint)
    summary = {"model": name, "checkpoint": str(checkpoint), "model_size_mb": checkpoint_size_mb(checkpoint), "metrics": metrics, "metadata": metadata}
    (work_dir / f"{name}.metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _save_existing_stage(work_dir: Path, name: str, checkpoint: Path, metrics: dict, metadata: dict) -> dict:
    """Record M0 without reserializing or retraining the accepted baseline."""
    summary = {"model": name, "checkpoint": str(checkpoint), "model_size_mb": checkpoint_size_mb(checkpoint), "metrics": metrics, "metadata": metadata}
    (work_dir / f"{name}.metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.qat_epochs < 1 or args.recovery_epochs < 1:
        raise ValueError("invalid loader or epoch arguments")
    if any(stage in {"M1I4", "M1I4KD", "M1I4FKD"} for stage in args.stages) and args.quant_bits != 4:
        raise ValueError("M1I4 stages require --quant-bits 4")
    root = Path(__file__).resolve().parents[3]
    _import_protogcn(root)
    _seed(args.seed)
    cfg = _load_config(Path(args.config))
    ann_file = Path(args.ann_file).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not ann_file.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("config, annotation and M0 checkpoint must exist")
    work_dir = Path(args.work_dir).resolve()
    all_ann = _all_annotation(ann_file, work_dir / "annotations_with_all.pkl")
    kd = LogitsKDConfig(args.temperature, args.ce_weight, args.kd_weight)
    feature_kd = FeatureKDConfig(weight=args.feature_kd_weight)
    pruning = StructuredPruningConfig(args.prune_ratio)
    contract = {"seed": args.seed, "splits": list(SPLITS), "ann_file": str(ann_file), "all_ann_file": str(all_ann), "checkpoint": str(checkpoint), "qat_epochs": args.qat_epochs, "recovery_epochs": args.recovery_epochs, "qat_lr": args.qat_lr, "recovery_lr": args.recovery_lr, "grad_clip_norm": args.grad_clip_norm, "quant_bits": args.quant_bits, "logits_kd": asdict(kd), "feature_kd": asdict(feature_kd), "structured_pruning": asdict(pruning)}
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "experiment_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **contract}, ensure_ascii=False))
        return

    train_loader = _train_loader(cfg, ann_file, args.batch_size, args.workers, args.seed)
    baseline = _build_model(cfg, checkpoint, args.device)
    summaries = {}
    if "M0" in args.stages:
        summaries["M0"] = _save_existing_stage(work_dir, "M0", checkpoint, evaluate_splits(baseline, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "read_only_M0", **contract})

    if "M1" in args.stages:
        m1 = copy.deepcopy(baseline)
        qat = WeightActivationFakeQuant(m1, args.quant_bits); qat.prepare()
        history = _train(m1, train_loader, args.qat_epochs, args.qat_lr, args.device)
        qat.remove()
        summaries["M1"] = _save_stage(work_dir, "M1", m1, evaluate_splits(m1, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_qat", "history": history, **contract}, quant_bits=args.quant_bits)

    if "M1KD" in args.stages:
        # Same unpruned M1 capacity: this isolates the effect of logits KD
        # from the already-measured effect of quantization and pruning.
        m1kd = copy.deepcopy(baseline)
        teacher = copy.deepcopy(baseline); freeze_teacher(teacher)
        qat = WeightActivationFakeQuant(m1kd, args.quant_bits); qat.prepare()
        history, best = _train_selecting_best_val(
            m1kd, train_loader, cfg, all_ann, args.qat_epochs, args.qat_lr,
            args.device, batch_size=args.batch_size, workers=args.workers,
            teacher=teacher, kd=kd,
        )
        qat.remove()
        summaries["M1KD"] = _save_stage(work_dir, "M1KD", m1kd, evaluate_splits(m1kd, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_teacher__M1_structure_qat_logits_kd", "qat_kd": history, "best_checkpoint": best, **contract}, quant_bits=args.quant_bits)

    if "M1FKD" in args.stages:
        # Identical unpruned M1 capacity: compare normalized Early/Middle/Deep
        # feature alignment against logits-only KD without a projection layer.
        m1fkd = copy.deepcopy(baseline)
        teacher = copy.deepcopy(baseline); freeze_teacher(teacher)
        aligner = ThreeStageFeatureAligner(teacher, m1fkd, feature_kd)
        qat = WeightActivationFakeQuant(m1fkd, args.quant_bits); qat.prepare()
        try:
            history, best = _train_selecting_best_val(
                m1fkd, train_loader, cfg, all_ann, args.qat_epochs, args.qat_lr,
                args.device, batch_size=args.batch_size, workers=args.workers,
                teacher=teacher, kd=kd, feature_aligner=aligner,
            )
        finally:
            aligner.close()
        qat.remove()
        summaries["M1FKD"] = _save_stage(work_dir, "M1FKD", m1fkd, evaluate_splits(m1fkd, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_teacher__M1_structure_qat_logits_feature_kd", "qat_kd": history, "best_checkpoint": best, **contract}, quant_bits=args.quant_bits)

    if "M1I4" in args.stages:
        m1i4 = copy.deepcopy(baseline)
        qat = WeightActivationFakeQuant(m1i4, num_bits=4); qat.prepare()
        history, best = _train_selecting_best_val(
            m1i4, train_loader, cfg, all_ann, args.qat_epochs, args.qat_lr,
            args.device, batch_size=args.batch_size, workers=args.workers,
            grad_clip_norm=args.grad_clip_norm,
        )
        qat.remove()
        summaries["M1I4"] = _save_stage(work_dir, "M1I4", m1i4, evaluate_splits(m1i4, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_int4_qat", "qat": history, "best_checkpoint": best, **contract}, quant_bits=4)

    if "M1I4KD" in args.stages:
        m1i4kd = copy.deepcopy(baseline)
        teacher = copy.deepcopy(baseline); freeze_teacher(teacher)
        qat = WeightActivationFakeQuant(m1i4kd, num_bits=4); qat.prepare()
        history, best = _train_selecting_best_val(
            m1i4kd, train_loader, cfg, all_ann, args.qat_epochs, args.qat_lr,
            args.device, batch_size=args.batch_size, workers=args.workers,
            teacher=teacher, kd=kd, grad_clip_norm=args.grad_clip_norm,
        )
        qat.remove()
        summaries["M1I4KD"] = _save_stage(work_dir, "M1I4KD", m1i4kd, evaluate_splits(m1i4kd, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_teacher__M1_structure_int4_qat_logits_kd", "qat_kd": history, "best_checkpoint": best, **contract}, quant_bits=4)

    if "M1I4FKD" in args.stages:
        m1i4fkd = copy.deepcopy(baseline)
        teacher = copy.deepcopy(baseline); freeze_teacher(teacher)
        aligner = ThreeStageFeatureAligner(teacher, m1i4fkd, feature_kd)
        qat = WeightActivationFakeQuant(m1i4fkd, num_bits=4); qat.prepare()
        try:
            history, best = _train_selecting_best_val(
                m1i4fkd, train_loader, cfg, all_ann, args.qat_epochs, args.qat_lr,
                args.device, batch_size=args.batch_size, workers=args.workers,
                teacher=teacher, kd=kd, feature_aligner=aligner,
                grad_clip_norm=args.grad_clip_norm,
            )
        finally:
            aligner.close()
        qat.remove()
        summaries["M1I4FKD"] = _save_stage(work_dir, "M1I4FKD", m1i4fkd, evaluate_splits(m1i4fkd, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_teacher__M1_structure_int4_qat_logits_feature_kd", "qat_kd": history, "best_checkpoint": best, **contract}, quant_bits=4)

    if "M2" in args.stages or "M3" in args.stages:
        m2_recovered, pruning_result = build_structurally_pruned_protogcn(
            cfg, baseline, args.device, pruning
        )
        recovery = _train(m2_recovered, train_loader, args.recovery_epochs, args.recovery_lr, args.device)
        if "M2" in args.stages:
            m2 = copy.deepcopy(m2_recovered)
            qat = WeightActivationFakeQuant(m2, args.quant_bits); qat.prepare()
            qat_history = _train(m2, train_loader, args.qat_epochs, args.qat_lr, args.device)
            qat.remove()
            summaries["M2"] = _save_stage(work_dir, "M2", m2, evaluate_splits(m2, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_structured_prune_qat", "pruning": pruning_result, "recovery": recovery, "qat": qat_history, **contract}, quant_bits=args.quant_bits)
        if "M3" in args.stages:
            m3 = copy.deepcopy(m2_recovered)
            teacher = copy.deepcopy(baseline); freeze_teacher(teacher)
            qat = WeightActivationFakeQuant(m3, args.quant_bits); qat.prepare()
            history = _train(m3, train_loader, args.qat_epochs, args.qat_lr, args.device, teacher=teacher, kd=kd)
            qat.remove()
            summaries["M3"] = _save_stage(work_dir, "M3", m3, evaluate_splits(m3, cfg, all_ann, device=args.device, batch_size=args.batch_size, workers=args.workers), {"source": "M0_teacher__M2_structure_qat_logits_kd", "pruning": pruning_result, "recovery": recovery, "qat_kd": history, **contract}, quant_bits=args.quant_bits)
    (work_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "work_dir": str(work_dir), "models": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
