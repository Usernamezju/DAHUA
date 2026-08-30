"""Architecture-safe structured channel pruning for the custom ProtoGCN."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuredPruningConfig:
    """Channel removal policy used by M2 and M3."""

    ratio: float = 0.40
    round_to: int = 8
    importance: str = "l2"

    def validate(self) -> None:
        if not 0 < self.ratio < 1:
            raise ValueError("structured pruning ratio must be in (0, 1)")
        if self.round_to < 1:
            raise ValueError("round_to must be positive")
        if self.importance != "l2":
            raise ValueError("only l2 channel importance is supported")


def _rounded_width(width: int, keep_ratio: float, multiple: int) -> int:
    candidate = int(round(width * keep_ratio / multiple) * multiple)
    return max(multiple, min(width - 1, candidate))


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _copy_magnitude_initialized_weights(source, target) -> int:
    """Transfer compatible slices, preferring high-L2 output channels.

    ProtoGCN's dynamic topology tensors are reshaped and concatenated in
    custom operators, so generic graph pruning cannot safely track every
    dependency.  A narrower ProtoGCN physically deletes feature channels.
    """
    import torch

    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred = 0
    with torch.no_grad():
        for name, destination in target_state.items():
            origin = source_state.get(name)
            if origin is None or origin.ndim != destination.ndim:
                continue
            if origin.ndim == 0:
                destination.copy_(origin)
                transferred += 1
                continue
            selected = origin
            if origin.shape[0] > destination.shape[0]:
                reduce_dims = tuple(range(1, origin.ndim))
                scores = origin.float().pow(2).sum(dim=reduce_dims) if reduce_dims else origin.float().pow(2)
                indices = scores.topk(destination.shape[0], largest=True).indices.sort().values
                selected = selected.index_select(0, indices)
            slices = tuple(slice(0, min(selected.shape[index], destination.shape[index])) for index in range(destination.ndim))
            destination.zero_()
            destination[slices].copy_(selected[slices])
            transferred += int(destination[slices].numel())
    target.load_state_dict(target_state, strict=True)
    return transferred


def build_structurally_pruned_protogcn(cfg, reference_model, device: str, config: StructuredPruningConfig):
    """Build a physically narrower, magnitude-initialized Campus6 ProtoGCN.

    All intermediate GCN/TCN widths are reduced by about 40% (rounded to a
    multiple of eight); the six-way output stays unchanged.  The resulting
    Conv/BN/Linear tensors are genuinely smaller, not zero-masked.
    """
    config.validate()
    from protogcn.models import build_recognizer

    base_width = int(reference_model.backbone.base_channels)
    # Convolutional parameter count scales approximately with C^2.  Use the
    # square root of the requested *parameter* reduction before rounding, so
    # a 40% target gives 72 rather than 56 base channels for Campus6.
    compact_width = _rounded_width(
        base_width, math.sqrt(1.0 - config.ratio), config.round_to
    )
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone["base_channels"] = compact_width
    ratio = float(model_cfg.backbone.get("ch_ratio", reference_model.backbone.ch_ratio))
    final_width = int(compact_width * ratio * ratio + 1e-4)
    model_cfg.cls_head["in_channels"] = final_width

    compact = build_recognizer(model_cfg)
    compact.to(device)
    transferred = _copy_magnitude_initialized_weights(reference_model, compact)
    result = {
        "config": asdict(config),
        "method": "width_rebuild_magnitude_transfer",
        "source_base_channels": base_width,
        "target_base_channels": compact_width,
        "source_final_channels": int(reference_model.cls_head.in_c),
        "target_final_channels": final_width,
        "base_params": _parameter_count(reference_model),
        "pruned_params": _parameter_count(compact),
        "transferred_elements": transferred,
        "architecture": {
            "backbone": {"base_channels": compact_width, "ch_ratio": ratio},
            "cls_head": {"in_channels": final_width},
        },
    }
    result["removed_param_fraction"] = 1.0 - result["pruned_params"] / max(result["base_params"], 1)
    return compact, result
