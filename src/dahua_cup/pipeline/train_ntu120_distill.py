"""Launch NTU120 ProtoGCN response distillation from an immutable checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dahua_cup.pipeline.common import render_command, require_file, run_command


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPOSITORY_ROOT
    / "dahua_cup"
    / "configs"
    / "protogcn"
    / "ntu120_ntu25_bone_distill.py"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-command",
        default=(
            "bash gcn_models/ProtoGCN/tools/dist_train.sh "
            "{config} {gpus} --validate"
        ),
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.gpus < 1 or args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("gpus/epochs must be positive and learning-rate > 0")
    annotation = require_file(args.ann_file, "NTU120 distillation annotation")
    checkpoint = require_file(args.init_checkpoint, "NTU120 initialization checkpoint")
    config = require_file(CONFIG, "NTU120 distillation config")
    work_dir = Path(args.work_dir).expanduser().resolve()
    command = render_command(
        args.train_command,
        config=config,
        gpus=args.gpus,
        ann_file=annotation,
        work_dir=work_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    environment = {
        "DAHUA_NTU120_DISTILL_ANN": str(annotation),
        "DAHUA_NTU120_DISTILL_WORK_DIR": str(work_dir),
        "DAHUA_PROTOGCN_NTU120_INIT": str(checkpoint),
        "DAHUA_NTU120_DISTILL_EPOCHS": str(args.epochs),
        "DAHUA_NTU120_DISTILL_LR": str(args.learning_rate),
        # dist_train.sh invokes ``python``. Put this interpreter first so a
        # Web process cannot accidentally launch training in another env.
        "PATH": str(Path(sys.executable).resolve().parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }
    run_command(command, dry_run=args.dry_run, env=environment)


if __name__ == "__main__":
    main()
