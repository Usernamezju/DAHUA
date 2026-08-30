"""Launch a reproducible ProtoGCN train job in its dedicated environment."""

from __future__ import annotations

import argparse

from .common import render_command, require_file, run_command


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--train-command",
        default="bash gcn_models/ProtoGCN/tools/dist_train.sh {config} 1 --validate",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = require_file(args.config, "ProtoGCN train config")
    run_command(render_command(args.train_command, config=config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
