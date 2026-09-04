"""Small, checkpoint-compatible LoRA adapters for Campus6 incremental runs.

The adapter is intentionally limited to the Campus6 classifier projection.
It keeps the ProtoGCN backbone and the pretrained classifier weight frozen,
then learns only a fixed-rank residual.  The wrapper translates legacy linear
checkpoint keys while loading, so an M0 FP32 checkpoint can initialise an
incremental LoRA run without a conversion step.
"""

from __future__ import annotations

import math
from typing import Iterable


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised on GPU server
        raise RuntimeError("LoRA incremental training requires PyTorch") from exc
    return torch, nn, functional


def _replace_child(root, dotted_name: str, value) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def install_classifier_lora(model, *, rank: int = 8, alpha: float = 16.0) -> list[str]:
    """Freeze ``model`` and place one LoRA adapter on ``cls_head.fc_cls``.

    A deliberately narrow target is auditable and stable across the shipped
    ProtoGCN architecture.  Returning the trainable parameter names makes the
    training log able to prove that no backbone parameter will update.
    """
    _torch_value, nn, _functional = _torch()
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    target = "cls_head.fc_cls"
    current = model.cls_head.fc_cls
    if getattr(current, "_dahua_lora", False):
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not isinstance(current, nn.Linear):
        raise TypeError("Campus6 LoRA target cls_head.fc_cls must be torch.nn.Linear")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _replace_child(model, target, LoRALinear(current, rank=rank, alpha=alpha))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if set(trainable) != {"cls_head.fc_cls.lora_down", "cls_head.fc_cls.lora_up"}:
        raise RuntimeError("Campus6 LoRA did not leave exactly its adapter trainable")
    return trainable


class LoRALinear:  # dynamically becomes an nn.Module when imported with torch
    """Placeholder base so importing this module stays light without torch."""

    def __new__(cls, base, *, rank: int, alpha: float):
        _torch_value, nn, functional = _torch()

        class _LoRALinear(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self._dahua_lora = True
                self.base = linear
                for parameter in self.base.parameters():
                    parameter.requires_grad_(False)
                self.rank = rank
                self.alpha = float(alpha)
                self.scaling = self.alpha / self.rank
                self.lora_down = nn.Parameter(_torch_value.empty(rank, linear.in_features))
                self.lora_up = nn.Parameter(_torch_value.zeros(linear.out_features, rank))
                nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))

            def forward(self, value):
                residual = functional.linear(functional.linear(value, self.lora_down), self.lora_up)
                return self.base(value) + residual * self.scaling

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                # M0 checkpoints use ``fc_cls.weight``/``bias``; adapters use
                # ``fc_cls.base.weight``/``bias``.  Translate without mutating
                # caller-owned checkpoint dictionaries.
                for name in ("weight", "bias"):
                    old = prefix + name
                    new = prefix + "base." + name
                    if old in state_dict and new not in state_dict:
                        state_dict[new] = state_dict.pop(old)
                super()._load_from_state_dict(
                    state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs,
                )

        return _LoRALinear(base)
