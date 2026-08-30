"""Quality-weighted hard, pseudo-label and logits distillation losses."""

from __future__ import annotations


def distillation_loss(
    student_logits,
    hard_targets,
    teacher_distribution,
    quality_weight,
    *,
    hard_label_mask=None,
    teacher_valid_mask=None,
    pseudo_label_mask=None,
    temperature=2.0,
    hard_weight=1.0,
    pseudo_weight=1.0,
    knowledge_weight=1.0,
):
    import torch
    import torch.nn.functional as functional

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = student_logits.shape[0]
    quality = torch.as_tensor(quality_weight, device=student_logits.device, dtype=student_logits.dtype)
    if quality.ndim == 0:
        quality = quality.expand(batch_size)
    quality = quality.reshape(batch_size).clamp(0, 1)
    hard_mask = torch.ones(
        batch_size, dtype=torch.bool, device=student_logits.device
    ) if hard_label_mask is None else torch.as_tensor(
        hard_label_mask, device=student_logits.device, dtype=torch.bool
    ).reshape(batch_size)
    teacher_mask = torch.ones(
        batch_size, dtype=torch.bool, device=student_logits.device
    ) if teacher_valid_mask is None else torch.as_tensor(
        teacher_valid_mask, device=student_logits.device, dtype=torch.bool
    ).reshape(batch_size)
    pseudo_mask = teacher_mask if pseudo_label_mask is None else torch.as_tensor(
        pseudo_label_mask, device=student_logits.device, dtype=torch.bool
    ).reshape(batch_size)
    hard_targets = torch.as_tensor(
        hard_targets, device=student_logits.device, dtype=torch.long
    ).reshape(batch_size)
    teacher = torch.as_tensor(teacher_distribution, device=student_logits.device, dtype=student_logits.dtype)
    if teacher.shape != student_logits.shape:
        raise ValueError("teacher_distribution must match student logits shape")
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pseudo_targets = teacher.argmax(dim=-1)

    zero = student_logits.sum() * 0.0
    hard = (
        functional.cross_entropy(student_logits[hard_mask], hard_targets[hard_mask])
        if hard_mask.any()
        else zero
    )
    pseudo_each = functional.cross_entropy(student_logits, pseudo_targets, reduction="none")
    pseudo_weight_value = quality * pseudo_mask
    pseudo = (
        (pseudo_each * pseudo_weight_value).sum()
        / pseudo_weight_value.sum().clamp_min(1.0)
        if pseudo_mask.any()
        else zero
    )
    student_log = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = functional.softmax(torch.log(teacher.clamp_min(1e-8)) / temperature, dim=-1)
    kl_each = functional.kl_div(student_log, teacher_soft, reduction="none").sum(dim=-1)
    knowledge_weight_value = quality * teacher_mask
    knowledge = (
        (kl_each * knowledge_weight_value).sum()
        / knowledge_weight_value.sum().clamp_min(1.0)
        * temperature ** 2
        if teacher_mask.any()
        else zero
    )
    total = hard_weight * hard + pseudo_weight * pseudo + knowledge_weight * knowledge
    return {"total": total, "hard": hard, "pseudo": pseudo, "knowledge": knowledge}


def weighted_distribution_kl(
    student_logits,
    target_distribution,
    sample_weight,
    valid_mask,
    *,
    temperature=2.0,
):
    """KL(target || student), used for previous-model retention on replay."""
    import torch
    import torch.nn.functional as functional

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = student_logits.shape[0]
    target = torch.as_tensor(
        target_distribution,
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    if target.shape != student_logits.shape:
        raise ValueError("target distribution must match student logits shape")
    weight = torch.as_tensor(
        sample_weight, device=student_logits.device, dtype=student_logits.dtype
    ).reshape(batch_size).clamp(0, 1)
    valid = torch.as_tensor(
        valid_mask, device=student_logits.device, dtype=torch.bool
    ).reshape(batch_size)
    effective = weight * valid
    if not valid.any():
        return student_logits.sum() * 0.0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_soft = functional.softmax(
        torch.log(target.clamp_min(1e-8)) / temperature, dim=-1
    )
    student_log = functional.log_softmax(student_logits / temperature, dim=-1)
    each = functional.kl_div(
        student_log, target_soft, reduction="none"
    ).sum(dim=-1)
    return (
        (each * effective).sum()
        / effective.sum().clamp_min(1.0)
        * temperature ** 2
    )
