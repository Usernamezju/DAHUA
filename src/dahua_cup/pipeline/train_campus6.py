"""Launch reproducible Campus6 ProtoGCN fine-tuning or response distillation."""

from __future__ import annotations

import argparse
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
        PROTOGCN_ROOT / "configs/campus6/rtmpose26_k400_2d_gap_full.py"
        if args.distill else PROTOGCN_ROOT / "configs/campus6/rtmpose26_k400_2d_full.py"
    )
    require_file(config, "Campus6 ProtoGCN config")
    environment = {}
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


if __name__ == "__main__":
    main()
