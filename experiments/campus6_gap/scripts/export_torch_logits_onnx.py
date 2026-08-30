#!/usr/bin/env python3
"""Export a supplied PyTorch logit module to dynamic-batch ONNX.

Framework-specific loading stays outside this small export boundary.  A model
factory receives ``model_args`` and must return a CPU ``nn.Module`` whose
``forward(N,C,T,V,M)`` returns exactly ``N,6`` logits.  Keeping that boundary
explicit prevents accidentally exporting a training-only loss or a multi-part
GAP output tuple.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch


def resolve_factory(spec: str):
    if ":" not in spec:
        raise SystemExit("--model-factory must be module.submodule:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory


class LogitOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, keypoint):
        output = self.model(keypoint)
        # GAP models return (classification_logits, global, head, arms, ...).
        return output[0] if isinstance(output, (tuple, list)) else output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-factory", required=True)
    parser.add_argument("--model-args-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="1,3,64,25,2", help="N,C,T,V,M example input")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 5:
        raise SystemExit("--shape must be N,C,T,V,M")
    model = resolve_factory(args.model_factory)(**json.loads(args.model_args_json.read_text(encoding="utf-8")))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise SystemExit("unexpected checkpoint keys: " + ", ".join(incompatible.unexpected_keys[:10]))
    model.eval()
    wrapper = LogitOnly(model).eval()
    example = torch.zeros(shape, dtype=torch.float32)
    with torch.no_grad():
        logits = wrapper(example)
    if logits.shape != (shape[0], 6):
        raise SystemExit(f"expected N,6 logits, got {tuple(logits.shape)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, example, args.output, input_names=["keypoint"], output_names=["logits"],
        dynamic_axes={"keypoint": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(args.output)


if __name__ == "__main__":
    main()
