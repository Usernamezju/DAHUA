"""Train only the Set-aware Group STN from frozen-YOLO track records."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from dahua_cup.pipeline.common import read_jsonl
from dahua_cup.stn.teacher_labels import validate_teacher_record


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stn"
    / "yolo_distillation.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--output-dir")
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--device",
        help="One PyTorch device, such as cuda:4 or cpu",
    )
    device.add_argument(
        "--devices",
        help="Comma-separated CUDA device indices for DataParallel",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("STN training configuration requires PyYAML") from exc
    source = Path(path).expanduser().resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != "stn_yolo_distillation.v1":
        raise ValueError("unsupported STN training configuration")
    for section in ("data", "model", "loss", "training"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"STN config is missing section: {section}")
    return value


def _apply_overrides(config: dict, args) -> dict:
    data = config["data"]
    training = config["training"]
    overrides = {
        "teacher_jsonl": args.teacher_jsonl,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    training_overrides = {
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "workers": args.workers,
    }
    for key, value in training_overrides.items():
        if value is not None:
            training[key] = value
    if args.device is not None:
        training["device"] = args.device
        training.pop("devices", None)
    elif args.devices is not None:
        training["devices"] = args.devices
    return config


def parse_cuda_devices(value) -> list[int]:
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    devices = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            device = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device index: {text}") from exc
        if device < 0:
            raise ValueError("CUDA device indices must be non-negative")
        if device in devices:
            raise ValueError(f"duplicate CUDA device index: {device}")
        devices.append(device)
    if not devices:
        raise ValueError("--devices must contain at least one CUDA index")
    return devices


def _to_device(batch: dict, device) -> tuple:
    video = batch["video"].to(device=device, non_blocking=True)
    targets = {
        key: batch[key].to(device=device, non_blocking=True)
        for key in ("boxes_xyxy", "presence", "confidence")
    }
    return video, targets


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_config(args.config), args)
    data_config = config["data"]
    training_config = config["training"]
    device_ids = parse_cuda_devices(training_config.get("devices"))
    configured_device = (
        f"cuda:{device_ids[0]}"
        if device_ids
        else str(training_config["device"])
    )
    teacher_path = Path(data_config["teacher_jsonl"]).expanduser().resolve()
    rows = read_jsonl(teacher_path)
    for row in rows:
        validate_teacher_record(row)
    if not rows:
        raise ValueError("YOLO teacher JSONL is empty")
    summary = {
        "teacher_jsonl": str(teacher_path),
        "records": len(rows),
        "output_dir": str(
            Path(training_config["output_dir"]).expanduser().resolve()
        ),
        "device": configured_device,
        "devices": device_ids,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return

    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError("STN training requires PyTorch and NumPy") from exc
    from dahua_cup.stn.dataset import (
        YOLOTrackDistillationDataset,
        split_teacher_records,
    )
    from dahua_cup.stn.losses import GroupSTNDistillationLoss
    from dahua_cup.stn.model import (
        SetAwareGroupSTN,
        trainable_parameter_count,
    )

    seed = int(training_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    if device_ids:
        available = torch.cuda.device_count()
        invalid = [value for value in device_ids if value >= available]
        if invalid:
            raise RuntimeError(
                "configured CUDA device indices are unavailable: "
                + ", ".join(map(str, invalid))
            )

    train_rows, validation_rows = split_teacher_records(
        rows,
        validation_fraction=float(data_config["validation_fraction"]),
        seed=seed,
    )
    common_dataset = {
        "clip_length": int(data_config["clip_length"]),
        "frame_stride": int(data_config["frame_stride"]),
        "image_size": int(data_config["image_size"]),
        "scale_factors": tuple(data_config["scale_factors"]),
        "seed": seed,
    }
    train_dataset = YOLOTrackDistillationDataset(
        train_rows,
        training=True,
        horizontal_flip_probability=float(
            data_config["horizontal_flip_probability"]
        ),
        **common_dataset,
    )
    validation_dataset = YOLOTrackDistillationDataset(
        validation_rows,
        training=False,
        horizontal_flip_probability=0.0,
        **common_dataset,
    )
    loader_options = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["workers"]),
        "pin_memory": device.type == "cuda",
    }
    sampler = None
    if bool(data_config.get("balanced_sampling", True)):
        strata = [
            (
                str(row.get("source_dataset") or "unknown").upper(),
                min(2, len(row.get("selected_track_ids") or [])),
            )
            for row in train_rows
        ]
        counts = Counter(strata)
        sample_weights = torch.as_tensor(
            [1.0 / counts[value] for value in strata],
            dtype=torch.double,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_options
    )

    base_model = SetAwareGroupSTN(**config["model"]).to(device)
    criterion = GroupSTNDistillationLoss(
        context_factor=float(config["model"]["context_factor"]),
        minimum_crop_fraction=float(
            config["model"]["minimum_crop_fraction"]
        ),
        **config["loss"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    if epochs < 1:
        raise ValueError("training epochs must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(
            str(Path(args.resume).expanduser().resolve()),
            map_location=device,
        )
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(
            checkpoint.get("best_validation_loss", best_validation)
        )
    model = base_model
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(
            base_model,
            device_ids=device_ids,
            output_device=device_ids[0],
        )

    output_dir = Path(training_config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            video, targets = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(video, apply_transform=False)
                loss_values = criterion(outputs, targets)
                loss = loss_values["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_config["gradient_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                video, targets = _to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(video, apply_transform=False)
                    loss_values = criterion(outputs, targets)
                validation_losses.append(
                    float(loss_values["total"].detach().cpu())
                )
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "train_loss": _mean(train_losses),
            "validation_loss": _mean(validation_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        checkpoint = {
            "schema_version": "set_aware_group_stn.v1",
            "epoch": epoch,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_loss": min(
                best_validation, metrics["validation_loss"]
            ),
            "config": config,
            "metrics": metrics,
            "trainable_parameters": trainable_parameter_count(base_model),
        }
        latest_path = output_dir / "latest.pt"
        temporary = output_dir / "latest.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(latest_path)
        if metrics["validation_loss"] < best_validation:
            best_validation = metrics["validation_loss"]
            best_temporary = output_dir / "best.pt.tmp"
            torch.save(checkpoint, best_temporary)
            best_temporary.replace(output_dir / "best.pt")

    report = {
        **summary,
        "schema_version": "set_aware_group_stn_training.v1",
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "trainable_parameters": trainable_parameter_count(base_model),
        "best_validation_loss": best_validation,
        "history": history,
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    temporary_report = output_dir / "training_report.json.tmp"
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(output_dir / "training_report.json")


if __name__ == "__main__":
    main()
