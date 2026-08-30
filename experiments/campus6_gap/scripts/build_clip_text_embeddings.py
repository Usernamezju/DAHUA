#!/usr/bin/env python3
"""Encode the approved Campus6 GAP prompt bank with frozen OpenAI CLIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PARTS = ("global", "head", "arms", "torso", "legs")
LABELS = ("normal_walk", "normal_run", "playful_chase", "playful_push", "conflict_chase", "conflict_push")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gap-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bank = json.loads(args.prompts.read_text(encoding="utf-8"))
    entries = {row["label"]: row for row in bank["classes"]}
    if tuple(entries) != LABELS:
        raise ValueError("prompt classes must use the fixed Campus6 label order")
    sys.path.insert(0, str(args.gap_root))
    import clip
    model, _ = clip.load(str(args.clip_checkpoint), device=args.device, jit=False)
    # GAP's bundled CLIP loader preserves FP16 checkpoint tensors.  The
    # server's PyTorch 1.10 attention path expects text activations and
    # weights to share a dtype, so use frozen FP32 text encoding explicitly.
    model.float().eval()
    vectors = []
    with torch.no_grad():
        for label in LABELS:
            row = entries[label]
            part_vectors = []
            for part in PARTS:
                text = row[part]
                vector = model.encode_text(clip.tokenize([text]).to(args.device)).float()
                part_vectors.append(F.normalize(vector, dim=-1)[0])
            vectors.append(torch.stack(part_vectors))
    output = F.normalize(torch.stack(vectors), dim=-1).cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print({"shape": list(output.shape), "output": str(args.output)})


if __name__ == "__main__":
    main()
