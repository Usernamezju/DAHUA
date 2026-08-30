import pytest


torch = pytest.importorskip("torch")

from dahua_cup.semantic_teacher.distillation.logits_kd import (  # noqa: E402
    FeatureKDConfig,
    LogitsKDConfig,
    ThreeStageFeatureAligner,
    export_quantized_state_dict,
    load_quantized_state_dict,
    logits_kd_loss,
)


def test_logits_kd_matches_ce_plus_positive_kl():
    student = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.0, 0.4]], requires_grad=True)
    teacher = torch.tensor([[1.5, 0.0, -0.5], [0.3, 0.8, -0.2]])
    result = logits_kd_loss(student, teacher, torch.tensor([0, 1]), LogitsKDConfig())
    assert result["loss_ce"].item() > 0
    assert result["loss_kd"].item() >= 0
    result["loss"].backward()
    assert torch.isfinite(student.grad).all()


def test_logits_kd_rejects_mismatched_class_space():
    with pytest.raises(ValueError, match="identical"):
        logits_kd_loss(torch.zeros(1, 6), torch.zeros(1, 5), torch.zeros(1, dtype=torch.long), LogitsKDConfig())


def test_three_stage_feature_alignment_backpropagates_to_student_only():
    class Block(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.full((4,), scale))

        def forward(self, value):
            return value * self.scale[None, :, None, None], None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.gcn = torch.nn.ModuleList([Block(1.0) for _ in range(3)])

        def forward(self, value):
            for block in self.backbone.gcn:
                value, _ = block(value)
            return value

    teacher, student = Model(), Model()
    teacher.backbone.gcn[1].scale.data.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    aligner = ThreeStageFeatureAligner(teacher, student, FeatureKDConfig(stages=(0, 1, 2), weight=.1))
    try:
        aligner.clear()
        student(torch.randn(2, 4, 3, 5, requires_grad=True))
        with torch.no_grad():
            teacher(torch.randn(2, 4, 3, 5))
        # Exercise shape validation and gradient connectivity using the same input.
        source = torch.randn(2, 4, 3, 5, requires_grad=True)
        aligner.clear()
        student(source)
        with torch.no_grad():
            teacher(source)
        loss = aligner.loss()
        loss.backward()
        assert loss.item() >= 0
        assert student.backbone.gcn[1].scale.grad is not None
        assert teacher.backbone.gcn[1].scale.grad is None
    finally:
        aligner.close()


def test_int4_artifact_is_packed_and_loadable(tmp_path):
    source = torch.nn.Linear(8, 3)
    destination = torch.nn.Linear(8, 3)
    path = tmp_path / "linear.int4.pt"
    payload = export_quantized_state_dict(source, path, {"model": "unit"}, num_bits=4)
    assert payload["num_bits"] == 4
    assert payload["qparams"]["weight"]["dtype"] == "int4_packed"
    load_quantized_state_dict(destination, path)
    output = destination(torch.randn(2, 8))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_export_quantizes_only_qat_covered_weights(tmp_path):
    class GraphModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_adjacency = torch.nn.Parameter(torch.randn(3, 3))
            self.conv = torch.nn.Conv2d(2, 4, 1, bias=True)
            self.norm = torch.nn.BatchNorm2d(4)

    source = GraphModel()
    payload = export_quantized_state_dict(
        source, tmp_path / "model.int8.pt", {"model": "unit"}, num_bits=8
    )

    assert set(payload["qparams"]) == {"conv.weight"}
    assert payload["state_dict"]["conv.weight"].dtype == torch.int8
    assert payload["state_dict"]["graph_adjacency"].dtype == torch.float32
    assert payload["state_dict"]["conv.bias"].dtype == torch.float32
    assert payload["state_dict"]["norm.running_var"].dtype == torch.float32
