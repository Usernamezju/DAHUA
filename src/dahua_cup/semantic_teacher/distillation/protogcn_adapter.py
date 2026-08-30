"""ProtoGCN registry adapter for masked Campus6 response distillation."""

from __future__ import annotations

import torch

from protogcn.models.builder import RECOGNIZERS
from protogcn.models.recognizers.recognizergcn import RecognizerGCN

from .losses import distillation_loss, weighted_distribution_kl


def _vector(value, batch_size, *, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(batch_size)


@RECOGNIZERS.register_module()
class DistillRecognizerGCN(RecognizerGCN):
    """RecognizerGCN with human, pseudo, Qwen and previous-model objectives."""

    def __init__(self, *args, distill_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_cfg = {
            "temperature": 2.0,
            "hard_weight": 1.0,
            "pseudo_weight": 0.5,
            "knowledge_weight": 1.0,
            "previous_weight": 0.5,
            "csc_weight": 0.2,
        }
        self.distill_cfg.update(distill_cfg or {})

    def forward_train(
        self,
        keypoint,
        label,
        teacher_distribution,
        quality_weight,
        has_hard_label,
        teacher_valid,
        previous_distribution=None,
        previous_valid=None,
        **kwargs,
    ):
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]
        feature, graph = self.extract_feat(keypoint)
        student_logits = self.cls_head(feature)
        batch_size = student_logits.shape[0]
        device = student_logits.device
        hard_mask = _vector(
            has_hard_label, batch_size, dtype=torch.bool, device=device
        )
        teacher_mask = _vector(
            teacher_valid, batch_size, dtype=torch.bool, device=device
        )
        targets = _vector(label, batch_size, dtype=torch.long, device=device)
        quality = _vector(
            quality_weight,
            batch_size,
            dtype=student_logits.dtype,
            device=device,
        )
        cfg = self.distill_cfg
        values = distillation_loss(
            student_logits,
            targets,
            teacher_distribution,
            quality,
            hard_label_mask=hard_mask,
            teacher_valid_mask=teacher_mask,
            pseudo_label_mask=teacher_mask & ~hard_mask,
            temperature=float(cfg["temperature"]),
            hard_weight=1.0,
            pseudo_weight=1.0,
            knowledge_weight=1.0,
        )
        losses = {
            "loss_hard": float(cfg["hard_weight"]) * values["hard"],
            "loss_pseudo": float(cfg["pseudo_weight"]) * values["pseudo"],
            "loss_qwen_kd": (
                float(cfg["knowledge_weight"]) * values["knowledge"]
            ),
        }
        if hard_mask.any() and float(cfg["csc_weight"]):
            csc = self.cls_head.csc_loss(
                graph[hard_mask],
                targets[hard_mask].detach(),
                student_logits[hard_mask].detach(),
            ).mean()
            losses["loss_csc"] = float(cfg["csc_weight"]) * csc
            losses["hard_top1_acc"] = (
                student_logits[hard_mask].argmax(dim=-1)
                == targets[hard_mask]
            ).float().mean()
        if previous_distribution is not None and previous_valid is not None:
            previous_mask = _vector(
                previous_valid, batch_size, dtype=torch.bool, device=device
            )
            losses["loss_previous_kd"] = float(cfg["previous_weight"]) * (
                weighted_distribution_kl(
                    student_logits,
                    previous_distribution,
                    torch.ones_like(quality),
                    previous_mask,
                    temperature=float(cfg["temperature"]),
                )
            )
        return losses
