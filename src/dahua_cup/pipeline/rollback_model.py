"""Atomically restore the previous Campus6 production model pointer."""

from __future__ import annotations

import argparse
import json

from dahua_cup.semantic_teacher.incremental.release_gate import ProductionPointer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-pointer", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    value = ProductionPointer(args.production_pointer).rollback(
        args.actor, args.reason
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
