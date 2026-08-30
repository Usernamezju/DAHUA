"""Strip NTU120 class-specific tensors before loading a checkpoint into Campus6."""

from __future__ import annotations

import argparse
from pathlib import Path


DROP_PREFIXES = (
    "cls_head.fc_cls.",
    "cls_head.csc_loss.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def campus6_backbone_state(state: dict) -> tuple[dict, list[str]]:
    """Remove DDP prefixes and every NTU class-head tensor."""
    kept = {}
    removed = []
    for original_name, value in state.items():
        name = (
            original_name[len("module.") :]
            if original_name.startswith("module.")
            else original_name
        )
        if name.startswith(DROP_PREFIXES):
            removed.append(original_name)
            continue
        if name in kept:
            raise ValueError(f"duplicate checkpoint tensor after normalization: {name}")
        kept[name] = value
    return kept, sorted(removed)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("checkpoint conversion requires the ProtoGCN torch environment") from exc
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    kept, removed = campus6_backbone_state(state)
    if not removed:
        raise ValueError("checkpoint contains no recognized ProtoGCN class-specific tensors")
    output = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    output["state_dict"] = kept
    metadata = dict(output.get("meta") or {})
    metadata.update(
        {
            "campus6_backbone_initialization": True,
            "source_checkpoint": str(source),
            "removed_class_tensors": removed,
        }
    )
    output["meta"] = metadata
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print(f"saved Campus6 initialization checkpoint: {destination}")
    print(f"removed {len(removed)} class-specific tensors")


if __name__ == "__main__":
    main()
