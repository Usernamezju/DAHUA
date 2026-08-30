"""Reusable QAT and raw-logit distillation primitives for Campus6 models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LogitsKDConfig:
    """Fixed, auditable defaults for the M3 comparison."""

    temperature: float = 4.0
    ce_weight: float = 1.0
    kd_weight: float = 1.0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.ce_weight < 0 or self.kd_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.ce_weight == 0 and self.kd_weight == 0:
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True)
class FeatureKDConfig:
    """Three representative ProtoGCN blocks for normalized feature KD."""

    stages: tuple[int, int, int] = (2, 5, 9)
    weight: float = 0.1

    def validate(self) -> None:
        if len(self.stages) != 3 or len(set(self.stages)) != 3:
            raise ValueError("feature KD requires exactly three distinct stages")
        if min(self.stages) < 0:
            raise ValueError("feature KD stages must be non-negative")
        if self.weight < 0:
            raise ValueError("feature KD weight must be non-negative")


def logits_kd_loss(student_logits, teacher_logits, targets, config: LogitsKDConfig):
    """Return CE + T² KL(softmax(teacher/T) || softmax(student/T))."""
    import torch.nn.functional as functional

    config.validate()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    temperature = float(config.temperature)
    ce = functional.cross_entropy(student_logits, targets)
    kd = functional.kl_div(
        functional.log_softmax(student_logits / temperature, dim=-1),
        functional.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature ** 2
    return {
        "loss": config.ce_weight * ce + config.kd_weight * kd,
        "loss_ce": ce,
        "loss_kd": kd,
    }


def freeze_teacher(model) -> None:
    """Freeze M0 once; the teacher must never receive optimizer updates."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class ThreeStageFeatureAligner:
    """Capture Early/Middle/Deep GCN block outputs and calculate feature KD.

    Teacher and student must have the same unpruned ProtoGCN structure.  Each
    captured tensor has shape ``[N*persons, C, T, V]`` and is L2-normalized in
    its channel dimension before MSE comparison.
    """

    def __init__(self, teacher, student, config: FeatureKDConfig) -> None:
        config.validate()
        self.config = config
        self.teacher_features = {}
        self.student_features = {}
        self._handles = []
        for name, model, target in (
            ("teacher", teacher, self.teacher_features),
            ("student", student, self.student_features),
        ):
            blocks = model.backbone.gcn
            if max(config.stages) >= len(blocks):
                raise ValueError(
                    f"{name} ProtoGCN exposes {len(blocks)} blocks, "
                    f"cannot capture stage {max(config.stages)}"
                )
            for stage in config.stages:
                self._handles.append(blocks[stage].register_forward_hook(
                    self._capture(target, stage)
                ))

    @staticmethod
    def _capture(destination, stage):
        def hook(_module, _inputs, output):
            feature = output[0] if isinstance(output, tuple) else output
            destination[stage] = feature
        return hook

    def clear(self) -> None:
        self.teacher_features.clear()
        self.student_features.clear()

    def loss(self):
        import torch
        import torch.nn.functional as functional

        values = []
        for stage in self.config.stages:
            teacher = self.teacher_features.get(stage)
            student = self.student_features.get(stage)
            if teacher is None or student is None:
                raise RuntimeError(f"missing captured feature at ProtoGCN stage {stage}")
            if teacher.shape != student.shape:
                raise ValueError(
                    f"feature shape mismatch at stage {stage}: "
                    f"teacher={tuple(teacher.shape)}, student={tuple(student.shape)}"
                )
            values.append(functional.mse_loss(
                functional.normalize(student, p=2, dim=1),
                functional.normalize(teacher.detach(), p=2, dim=1),
            ))
        return torch.stack(values).mean()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class WeightActivationFakeQuant:
    """Lightweight fake-integer controller for arbitrary Conv/Linear graphs.

    It fake-quantizes every Conv2d/Linear weight through PyTorch
    parametrization and attaches output hooks to quantize their activations.
    The wrapped network therefore keeps normal floating-point operators during
    QAT while both weights and activations see the INT8 quantization error.
    """

    def __init__(self, model, num_bits: int = 8) -> None:
        if not 2 <= num_bits <= 8:
            raise ValueError("QAT bit width must be in [2, 8]")
        self.model = model
        self.num_bits = int(num_bits)
        self._handles = []
        self._parametrized = []

    def _fake_quant(self, value):
        import torch

        quant_max = 2 ** (self.num_bits - 1) - 1
        quant_min = -(2 ** (self.num_bits - 1))
        maximum = value.detach().abs().amax()
        scale = torch.clamp(maximum / quant_max, min=torch.finfo(value.dtype).eps)
        return torch.fake_quantize_per_tensor_affine(
            value, scale=float(scale), zero_point=0,
            quant_min=quant_min, quant_max=quant_max,
        )

    def prepare(self) -> None:
        import torch.nn as nn
        from torch.nn.utils import parametrize

        class _WeightFakeQuant(nn.Module):
            def __init__(self, num_bits):
                super().__init__()
                self.num_bits = num_bits

            def forward(self, weight):
                import torch

                quant_max = 2 ** (self.num_bits - 1) - 1
                quant_min = -(2 ** (self.num_bits - 1))
                maximum = weight.detach().abs().amax()
                scale = torch.clamp(maximum / quant_max, min=torch.finfo(weight.dtype).eps)
                return torch.fake_quantize_per_tensor_affine(
                    weight, scale=float(scale), zero_point=0,
                    quant_min=quant_min, quant_max=quant_max,
                )

        def output_hook(_module, _inputs, output):
            return self._fake_quant(output) if hasattr(output, "dtype") else output

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if not parametrize.is_parametrized(module, "weight"):
                    parametrize.register_parametrization(
                        module, "weight", _WeightFakeQuant(self.num_bits)
                    )
                    self._parametrized.append(module)
                self._handles.append(module.register_forward_hook(output_hook))

    def remove(self) -> None:
        """Materialize fake-quantized weights before checkpoint export."""
        from torch.nn.utils import parametrize

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module in self._parametrized:
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )
        self._parametrized.clear()


def export_quantized_state_dict(model, destination: str | Path, metadata: dict, num_bits: int = 8) -> dict:
    """Persist every floating tensor as per-tensor symmetric INT8 or INT4.

    This is a portable deployment artifact.  The paired loader dequantizes it
    for PyTorch evaluation; deployment backends may consume the int8 tensors
    directly or convert the model to ONNX QDQ.
    """
    import torch

    if num_bits not in (4, 8):
        raise ValueError("export supports only INT4 and INT8")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors, qparams = {}, {}
    for name, value in model.state_dict().items():
        if not value.is_floating_point():
            tensors[name] = value.cpu()
            continue
        cpu = value.detach().cpu().float()
        quant_max = 2 ** (num_bits - 1) - 1
        quant_min = -(2 ** (num_bits - 1))
        scale = max(float(cpu.abs().max()) / quant_max, 1e-12)
        quantized = torch.clamp(torch.round(cpu / scale), quant_min, quant_max).to(torch.int8)
        if num_bits == 4:
            flat = (quantized.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
            if flat.numel() % 2:
                flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
            tensors[name] = flat[0::2] | (flat[1::2] << 4)
            qparams[name] = {"scale": scale, "zero_point": 0, "dtype": "int4_packed", "shape": list(cpu.shape)}
        else:
            tensors[name] = quantized
            qparams[name] = {"scale": scale, "zero_point": 0, "dtype": "int8"}
    payload = {
        "format": "dahua.quantized_state.v2",
        "num_bits": num_bits,
        "metadata": metadata,
        "qparams": qparams,
        "state_dict": tensors,
    }
    torch.save(payload, destination)
    return payload


def export_int8_state_dict(model, destination: str | Path, metadata: dict) -> dict:
    """Backward-compatible INT8 export wrapper."""
    return export_quantized_state_dict(model, destination, metadata, num_bits=8)


def load_quantized_state_dict(model, source: str | Path):
    """Load a portable INT8/INT4 artifact into a floating evaluation model."""
    import torch

    payload = torch.load(source, map_location="cpu")
    if payload.get("format") not in {"dahua.int8_state.v1", "dahua.quantized_state.v2"}:
        raise ValueError("unsupported INT8 artifact")
    state = {}
    for name, value in payload["state_dict"].items():
        params = payload["qparams"].get(name)
        if not params:
            state[name] = value
        elif params["dtype"] == "int4_packed":
            packed = value.reshape(-1)
            unpacked = torch.stack((packed & 0x0F, packed >> 4), dim=1).reshape(-1)
            count = 1
            for dimension in params["shape"]:
                count *= dimension
            state[name] = (unpacked[:count].to(torch.int16) - 8).reshape(params["shape"]).float() * params["scale"]
        else:
            state[name] = value.float() * params["scale"]
    model.load_state_dict(state, strict=True)
    return dict(payload.get("metadata") or {})


def load_int8_state_dict(model, source: str | Path):
    """Backward-compatible alias for the generic quantized loader."""
    return load_quantized_state_dict(model, source)


def checkpoint_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1024.0 / 1024.0


def config_dict(config: LogitsKDConfig) -> dict:
    return asdict(config)
