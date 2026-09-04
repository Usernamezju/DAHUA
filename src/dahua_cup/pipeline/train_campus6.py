"""Launch reproducible Campus6 ProtoGCN fine-tuning or response distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dahua_cup.pipeline.common import render_command, require_file, run_command
from dahua_cup.paths import PROTOGCN_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument(
        "--lora", action="store_true",
        help="freeze ProtoGCN/base classifier weights and train the rank-8 LoRA adapter",
    )
    compression.add_argument(
        "--deep-compression", action="store_true",
        help=(
            "after fine-tuning, run Han et al. Deep Compression: 85%% pruning, "
            "8-bit trained weight sharing, distillation recovery and Huffman packing"
        ),
    )
    parser.add_argument("--deep-sparsity", type=float, default=0.85)
    parser.add_argument("--deep-weight-bits", type=int, default=8, choices=range(2, 9))
    parser.add_argument("--deep-prune-epochs", type=int, default=12)
    parser.add_argument("--deep-sharing-epochs", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-command",
        default=(
            "bash third_party/ProtoGCN/tools/dist_train.sh "
            "{config} {gpus} --validate --ann-file {ann_file} "
            "--work-dir {work_dir}{overrides}"
        ),
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.gpus < 1:
        raise ValueError("gpus must be positive")
    annotation = require_file(args.ann_file, "Campus6 annotation")
    config = (
        PROTOGCN_ROOT / "configs/campus6/rtm_s_coco17_k400_2d_gap_full.py"
        if args.distill else PROTOGCN_ROOT / "configs/campus6/rtm_s_coco17_k400_2d_full.py"
    )
    require_file(config, "Campus6 ProtoGCN config")
    environment = {}
    if args.lora:
        environment["DAHUA_PROTOGCN_LORA"] = "1"
    if args.init_checkpoint:
        environment["DAHUA_PROTOGCN_CAMPUS6_INIT"] = str(
            require_file(args.init_checkpoint, "Campus6 initialization checkpoint")
        )
    overrides = ""
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        overrides += f" --epochs {args.epochs}"
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        overrides += f" --learning-rate {args.learning_rate}"
    command = render_command(
        args.train_command,
        config=config,
        gpus=args.gpus,
        ann_file=annotation,
        work_dir=Path(args.work_dir).expanduser().resolve(),
        overrides=overrides,
    )
    run_command(command, dry_run=args.dry_run, env=environment)
    if not args.deep_compression:
        return
    if not 0 < args.deep_sparsity < 1:
        raise ValueError("deep_sparsity must be in (0, 1)")
    if args.deep_prune_epochs < 1 or args.deep_sharing_epochs < 1:
        raise ValueError("Deep Compression recovery epochs must be positive")
    work_dir = Path(args.work_dir).expanduser().resolve()
    if args.dry_run:
        trained_checkpoint = work_dir / "best_top1_acc_epoch_<N>.pth"
    else:
        candidates = sorted(
            work_dir.glob("best_top1_acc*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "fine-tuning completed without a best_top1_acc checkpoint; "
                "Deep Compression was not started"
            )
        trained_checkpoint = candidates[0]
    deep_command = [
        sys.executable,
        "-m", "dahua_cup.semantic_teacher.distillation.campus6_deep_compression",
        "--config", str(config), "--checkpoint", str(trained_checkpoint),
        "--ann-file", str(annotation), "--work-dir", str(work_dir / "deep_compression"),
        "--sparsity", str(args.deep_sparsity),
        "--conv-bits", str(args.deep_weight_bits),
        "--head-bits", str(args.deep_weight_bits),
        "--prune-epochs", str(args.deep_prune_epochs),
        "--sharing-epochs", str(args.deep_sharing_epochs),
    ]
    # The training interpreter contains MMCV/ProtoGCN; invoking it directly
    # avoids accidentally switching to the Web service's Python environment.
    if args.dry_run:
        deep_command.append("--dry-run")
    run_command(deep_command, dry_run=args.dry_run, env=environment)


if __name__ == "__main__":
    main()
