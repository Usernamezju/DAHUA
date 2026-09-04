import pytest


torch = pytest.importorskip("torch")

from dahua_cup.semantic_teacher.incremental.lora import install_classifier_lora


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_cls = torch.nn.Linear(4, 3)


class _CampusModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.cls_head = _Head()


def test_classifier_lora_freezes_base_and_loads_legacy_linear_checkpoint():
    baseline = _CampusModel()
    checkpoint = baseline.state_dict()
    expected = baseline.cls_head.fc_cls(torch.randn(2, 4))

    candidate = _CampusModel()
    trainable = install_classifier_lora(candidate, rank=2, alpha=4)
    candidate.load_state_dict(checkpoint, strict=False)

    assert trainable == [
        "cls_head.fc_cls.lora_down", "cls_head.fc_cls.lora_up",
    ]
    assert not candidate.backbone.weight.requires_grad
    assert not candidate.cls_head.fc_cls.base.weight.requires_grad
    # lora_up starts at zero, so legacy M0 logits are preserved before the
    # first incremental optimisation step.
    value = torch.randn(2, 4)
    assert torch.allclose(
        candidate.cls_head.fc_cls(value),
        baseline.cls_head.fc_cls(value),
    )
    assert expected.shape == (2, 3)
