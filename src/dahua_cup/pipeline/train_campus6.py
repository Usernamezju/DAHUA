"""Launch reproducible Campus6 ProtoGCN fine-tuning or response distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dahua_cup.pipeline.common import render_command, require_file, run_command


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "protogcn"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--backend", choices=("rtmpose17", "ntu25"), default="ntu25")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-command",
        default=(
            "bash gcn_models/ProtoGCN/tools/dist_train.sh "
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
    if args.backend == "rtmpose17":
        config = CONFIG_ROOT / "campus6_rtmpose17_distill.py" if args.distill else (
            Path(__file__).resolve().parents[2] / "gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_full.py"
        )
    else:
        config = CONFIG_ROOT / ("campus6_ntu25_bone_distill.py" if args.distill else "campus6_ntu25_bone.py")
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
