"""Export a weight-only checkpoint suitable for inference and size gates."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("checkpoint export requires torch") from exc
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(str(source), map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("input checkpoint has no state_dict")
    output = {
        "state_dict": checkpoint["state_dict"],
        "meta": dict(checkpoint.get("meta") or {}),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, str(temporary))
    temporary.replace(destination)
    print(
        f"exported {destination} ({destination.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
