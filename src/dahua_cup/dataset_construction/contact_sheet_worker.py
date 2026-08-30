"""Subprocess entry point for crash-isolated RGB contact-sheet generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contact_sheets import create_contact_sheets


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config_json)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to an object")

    output_dir = Path(args.output_dir)
    sheets, metadata = create_contact_sheets(Path(args.video), output_dir, config)
    if not sheets:
        raise ValueError("contact-sheet generation returned no sheets")
    _atomic_json(output_dir / "metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
