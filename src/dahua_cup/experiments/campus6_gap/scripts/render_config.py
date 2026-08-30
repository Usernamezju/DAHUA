#!/usr/bin/env python3
"""Render an experiment config from an audited path-free template."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    values = dict(os.environ)
    for assignment in args.set:
        if "=" not in assignment:
            raise SystemExit("--set must have NAME=VALUE form")
        name, value = assignment.split("=", 1)
        values[name] = value
    source = args.template.read_text(encoding="utf-8")
    missing = sorted({name for name in PATTERN.findall(source) if not values.get(name)})
    if missing:
        raise SystemExit("missing values for: " + ", ".join(missing))
    rendered = PATTERN.sub(lambda match: values[match.group(1)], source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
