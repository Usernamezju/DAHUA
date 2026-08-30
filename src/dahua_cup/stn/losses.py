"""Losses for two-query YOLO localization distillation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import square_group_boxes, xyxy_to_cxcywh


def aligned_generalized_iou(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Generalized IoU for aligned normalized xyxy boxes."""
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("aligned boxes must have equal [...,4] shapes")
    intersection_left_top = torch.maximum(first[..., :2], second[..., :2])
    intersection_right_bottom = torch.minimum(
        first[..., 2:], second[..., 2:]
    )
    intersection_size = (
        intersection_right_bottom - intersection_left_top
    ).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)
    first_size = (first[..., 2:] - first[..., :2]).clamp_min(0)
    second_size = (second[..., 2:] - second[..., :2]).clamp_min(0)
    union = (
        first_size.prod(dim=-1)
        + second_size.prod(dim=-1)
        - intersection
    ).clamp_min(1e-7)
    iou = intersection / union
    enclosing_left_top = torch.minimum(first[..., :2], second[..., :2])
    enclosing_right_bottom = torch.maximum(first[..., 2:], second[..., 2:])
    enclosing = (
        enclosing_right_bottom - enclosing_left_top
    ).clamp_min(0).prod(dim=-1).clamp_min(1e-7)
    return iou - (enclosing - union) / enclosing


def target_group_boxes(
    boxes: torch.Tensor,
    presence: torch.Tensor,
    *,
    context_factor: float,
    minimum_fraction: float,
) -> torch.Tensor:
    """Merge teacher person boxes; an empty frame maps to full-frame."""
    active = presence.to(dtype=torch.bool)
    left_values = boxes[..., 0]
    top_values = boxes[..., 1]
    right_values = boxes[..., 2]
    bottom_values = boxes[..., 3]
    left = torch.where(
        active, left_values, torch.ones_like(left_values)
    ).amin(dim=2)
    top = torch.where(
        active, top_values, torch.ones_like(top_values)
    ).amin(dim=2)
    right = torch.where(
        active, right_values, torch.zeros_like(right_values)
    ).amax(dim=2)
    bottom = torch.where(
        active, bottom_values, torch.zeros_like(bottom_values)
    ).amax(dim=2)
    merged = torch.stack((left, top, right, bottom), dim=-1)
    full_frame = torch.zeros_like(merged)
    full_frame[..., 2:] = 1.0
    merged = torch.where(
        active.any(dim=2)[..., None], merged, full_frame
    )
    return square_group_boxes(
        merged,
        context_factor=context_factor,
        minimum_fraction=minimum_fraction,
    )


def _assignment_cost(
    predicted_boxes: torch.Tensor,
    presence_logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_presence: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    target_values = target_presence.to(dtype=presence_logits.dtype)
    presence_cost = F.binary_cross_entropy_with_logits(
        presence_logits, target_values, reduction="none"
    ).mean(dim=(1, 2))
    weights = target_values * confidence.clamp(0.0, 1.0)
    normalizer = weights.sum(dim=(1, 2)).clamp_min(1.0)
    l1 = (
        (predicted_boxes - target_boxes).abs().sum(dim=-1) * weights
    ).sum(dim=(1, 2)) / normalizer
    giou = (
        (1.0 - aligned_generalized_iou(predicted_boxes, target_boxes))
        * weights
    ).sum(dim=(1, 2)) / normalizer
    return presence_cost + l1 + giou


def match_two_queries(
    predicted_boxes: torch.Tensor,
    presence_logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_presence: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact clip-level set matching by evaluating the two permutations."""
    expected = predicted_boxes.shape[:-1]
    if (
        predicted_boxes.ndim != 4
        or predicted_boxes.shape[2] != 2
        or target_boxes.shape != predicted_boxes.shape
        or presence_logits.shape != expected
        or target_presence.shape != expected
        or confidence.shape != expected
    ):
        raise ValueError("set matching expects aligned [B,T,2] targets")
    identity = _assignment_cost(
        predicted_boxes,
        presence_logits,
        target_boxes,
        target_presence,
        confidence,
    )
    swapped_boxes = target_boxes.flip(dims=(2,))
    swapped_presence = target_presence.flip(dims=(2,))
    swapped_confidence = confidence.flip(dims=(2,))
    swapped = _assignment_cost(
        predicted_boxes,
        presence_logits,
        swapped_boxes,
        swapped_presence,
        swapped_confidence,
    )
    use_swap = swapped < identity
    selector_boxes = use_swap[:, None, None, None]
    selector_values = use_swap[:, None, None]
    return (
        torch.where(selector_boxes, swapped_boxes, target_boxes),
        torch.where(selector_values, swapped_presence, target_presence),
        torch.where(selector_values, swapped_confidence, confidence),
        use_swap,
    )


def rasterize_person_boxes(
    boxes: torch.Tensor,
    presence: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Rasterize the union of teacher boxes as ``[B,1,T,H,W]``."""
    if height < 1 or width < 1:
        raise ValueError("occupancy dimensions must be positive")
    x = (
        torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / width
    y = (
        torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / height
    x = x[None, None, None, None, :]
    y = y[None, None, None, :, None]
    active = presence.to(dtype=torch.bool)[..., None, None]
    inside_x = (
        (x >= boxes[..., 0, None, None])
        & (x <= boxes[..., 2, None, None])
    )
    inside_y = (
        (y >= boxes[..., 1, None, None])
        & (y <= boxes[..., 3, None, None])
    )
    occupancy = (inside_x & inside_y & active).any(dim=2)
    return occupancy[:, None].to(dtype=boxes.dtype)


class GroupSTNDistillationLoss(nn.Module):
    """Response-level YOLO-to-STN localization distillation."""

    def __init__(
        self,
        *,
        context_factor: float = 1.25,
        minimum_crop_fraction: float = 0.25,
        presence_weight: float = 1.0,
        box_weight: float = 5.0,
        giou_weight: float = 2.0,
        group_weight: float = 3.0,
        containment_weight: float = 4.0,
        occupancy_weight: float = 1.0,
        temporal_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.context_factor = context_factor
        self.minimum_crop_fraction = minimum_crop_fraction
        self.weights = {
            "presence": presence_weight,
            "box": box_weight,
            "giou": giou_weight,
            "group": group_weight,
            "containment": containment_weight,
            "occupancy": occupancy_weight,
            "temporal": temporal_weight,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        predicted_boxes = outputs["boxes_xyxy"]
        presence_logits = outputs["presence_logits"]
        target_boxes = targets["boxes_xyxy"].to(
            device=predicted_boxes.device, dtype=predicted_boxes.dtype
        )
        target_presence = targets["presence"].to(
            device=predicted_boxes.device, dtype=torch.bool
        )
        confidence = targets["confidence"].to(
            device=predicted_boxes.device, dtype=predicted_boxes.dtype
        )
        (
            matched_boxes,
            matched_presence,
            matched_confidence,
            swapped,
        ) = match_two_queries(
            predicted_boxes,
            presence_logits,
            target_boxes,
            target_presence,
            confidence,
        )
        target_values = matched_presence.to(dtype=predicted_boxes.dtype)
        positive_weights = (
            target_values * matched_confidence.clamp(0.0, 1.0)
        )
        box_normalizer = positive_weights.sum().clamp_min(1.0)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, target_values
        )
        box_loss = (
            (predicted_boxes - matched_boxes).abs().sum(dim=-1)
            * positive_weights
        ).sum() / box_normalizer
        giou_loss = (
            (1.0 - aligned_generalized_iou(
                predicted_boxes, matched_boxes
            ))
            * positive_weights
        ).sum() / box_normalizer

        teacher_group = target_group_boxes(
            matched_boxes,
            matched_presence,
            context_factor=self.context_factor,
            minimum_fraction=self.minimum_crop_fraction,
        )
        predicted_group = outputs["group_boxes_xyxy"]
        has_person = matched_presence.any(dim=2).to(
            dtype=predicted_boxes.dtype
        )
        group_normalizer = has_person.sum().clamp_min(1.0)
        group_l1 = (
            (predicted_group - teacher_group).abs().sum(dim=-1)
            * has_person
        ).sum() / group_normalizer
        group_giou = (
            (1.0 - aligned_generalized_iou(
                predicted_group, teacher_group
            ))
            * has_person
        ).sum() / group_normalizer
        group_loss = group_l1 + group_giou

        crop = predicted_group[:, :, None, :]
        containment = (
            F.relu(crop[..., 0] - matched_boxes[..., 0])
            + F.relu(crop[..., 1] - matched_boxes[..., 1])
            + F.relu(matched_boxes[..., 2] - crop[..., 2])
            + F.relu(matched_boxes[..., 3] - crop[..., 3])
        )
        containment_loss = (
            containment * positive_weights
        ).sum() / box_normalizer

        occupancy_logits = outputs["occupancy_logits"]
        occupancy_target = rasterize_person_boxes(
            matched_boxes,
            matched_presence,
            height=occupancy_logits.shape[-2],
            width=occupancy_logits.shape[-1],
        )
        occupancy_bce = F.binary_cross_entropy_with_logits(
            occupancy_logits, occupancy_target, reduction="none"
        )
        occupancy_probability = torch.sigmoid(occupancy_logits)
        probability_correct = torch.where(
            occupancy_target > 0.5,
            occupancy_probability,
            1.0 - occupancy_probability,
        )
        alpha = torch.where(
            occupancy_target > 0.5,
            torch.full_like(occupancy_target, 0.75),
            torch.full_like(occupancy_target, 0.25),
        )
        occupancy_loss = (
            alpha * (1.0 - probability_correct).pow(2) * occupancy_bce
        ).mean()

        predicted_motion = (
            xyxy_to_cxcywh(predicted_group[:, 1:])
            - xyxy_to_cxcywh(predicted_group[:, :-1])
        )
        teacher_motion = (
            xyxy_to_cxcywh(teacher_group[:, 1:])
            - xyxy_to_cxcywh(teacher_group[:, :-1])
        )
        temporal_mask = (
            has_person[:, 1:] * has_person[:, :-1]
        )
        temporal_loss = (
            F.smooth_l1_loss(
                predicted_motion, teacher_motion, reduction="none"
            ).sum(dim=-1)
            * temporal_mask
        ).sum() / temporal_mask.sum().clamp_min(1.0)

        components = {
            "presence": presence_loss,
            "box": box_loss,
            "giou": giou_loss,
            "group": group_loss,
            "containment": containment_loss,
            "occupancy": occupancy_loss,
            "temporal": temporal_loss,
        }
        total = sum(
            self.weights[name] * value
            for name, value in components.items()
        )
        return {
            "total": total,
            **components,
            "assignment_swap_ratio": swapped.to(
                dtype=predicted_boxes.dtype
            ).mean(),
        }
