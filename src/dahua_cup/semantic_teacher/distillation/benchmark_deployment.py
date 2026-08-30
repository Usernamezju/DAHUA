"""Benchmark comparable Campus6 FP32 and portable-INT8 deployment artifacts.

The project INT8 checkpoints store quantized tensors plus scales.  PyTorch
loads them into the same deployment model as FP32, so this benchmark records
the real execution mode instead of implying TensorRT/native-INT8 acceleration
where no such backend is in use.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .export_deployment_model import _model, _register_protogcn
from .logits_kd import load_quantized_state_dict


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fp32", required=True, help="M0 .pth checkpoint")
    value.add_argument("--int8", required=True, help="fine-tuned portable INT8 checkpoint")
    value.add_argument("--source-config", required=True, help="GAP training config")
    value.add_argument("--deploy-config", required=True, help="plain RecognizerGCN config")
    value.add_argument("--fp32-deployment", help="optional exported FP32 deployment artifact")
    value.add_argument("--int8-deployment", help="optional exported portable-INT8 deployment artifact")
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--warmup", type=int, default=50)
    value.add_argument("--iterations", type=int, default=300)
    return value


def _gpu_info(device: str) -> dict:
    import torch

    index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(index)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_index = int(visible.split(",")[index]) if visible else index
    values = {
        "logical_index": index,
        "physical_index": physical_index,
        "name": properties.name,
        "total_memory_mib": properties.total_memory / 1024.0 / 1024.0,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        values["driver_version"] = text[physical_index]
    except (OSError, subprocess.CalledProcessError, IndexError):
        values["driver_version"] = "unavailable"
    return values


def _load_deployment(source: Path, source_config: Path, deploy_config: Path, device: str):
    """Load source state, then retain only inference model weights."""
    from mmcv.runner import load_checkpoint

    source_model = _model(source_config, device, force_plain=False)
    if source.suffix == ".pth":
        load_checkpoint(source_model, str(source), map_location="cpu", strict=False)
        checkpoint_format = "fp32_checkpoint"
    else:
        metadata = load_quantized_state_dict(source_model, source)
        checkpoint_format = str(metadata.get("format", "portable_int8_state"))
    deploy_model = _model(deploy_config, device, force_plain=True)
    source_state = source_model.state_dict()
    deploy_state = deploy_model.state_dict()
    missing = [name for name in deploy_state if name not in source_state]
    if missing:
        raise RuntimeError(f"missing deployment weights: {missing[:3]}")
    deploy_model.load_state_dict({name: source_state[name] for name in deploy_state}, strict=True)
    del source_state, source_model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    deploy_model.eval()
    return deploy_model, checkpoint_format


def _forward(model, value):
    feature, _graph = model.extract_feat(value)
    return model.cls_head(feature)


def _summarize(samples_ms: list[float]) -> dict:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(values.std(ddof=1)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_samples_per_s": float(1000.0 / values.mean()),
    }


def _benchmark(name: str, checkpoint: Path, args, input_tensor) -> dict:
    import torch

    model, checkpoint_format = _load_deployment(
        checkpoint, Path(args.source_config), Path(args.deploy_config), args.device
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            _forward(model, input_tensor)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(args.device)
        resident = torch.cuda.memory_allocated(args.device)
        samples_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            _forward(model, input_tensor)
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        peak = torch.cuda.max_memory_allocated(args.device)
    result = {
        "name": name,
        "source_training_checkpoint": str(checkpoint),
        "source_training_checkpoint_bytes": checkpoint.stat().st_size,
        "source_training_checkpoint_size_mib": checkpoint.stat().st_size / 1024.0 / 1024.0,
        "checkpoint_format": checkpoint_format,
        "runtime_execution": "FP32 CUDA operators after checkpoint load",
        "latency": _summarize(samples_ms),
        "memory": {
            "resident_allocated_mib": resident / 1024.0 / 1024.0,
            "peak_allocated_mib": peak / 1024.0 / 1024.0,
            "incremental_inference_peak_mib": (peak - resident) / 1024.0 / 1024.0,
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None) -> None:
    import torch

    args = parser().parse_args(argv)
    if args.warmup < 1 or args.iterations < 10:
        raise ValueError("warmup must be >= 1 and iterations >= 10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    root = Path(__file__).resolve().parents[3]
    _register_protogcn(root)
    torch.manual_seed(20260829)
    input_tensor = torch.randn((1, 2, 100, 20, 3), device=args.device)
    result = {
        "schema_version": "campus6_deployment_benchmark.v1",
        "protocol": {
            "input_shape": list(input_tensor.shape),
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing": "host perf_counter around each CUDA forward plus synchronize",
            "memory": "PyTorch per-process CUDA allocated bytes; reset after warmup with model and input resident",
            "comparability": "same plain deployment graph and input for M0 FP32 and M1FKD portable INT8",
            "caveat": "portable INT8 tensors are dequantized at load for this PyTorch runtime; no native INT8 backend is installed",
        },
        "gpu": _gpu_info(args.device),
        "models": [
            _benchmark("M0_FP32", Path(args.fp32), args, input_tensor),
            _benchmark("M1FKD_portable_INT8", Path(args.int8), args, input_tensor),
        ],
    }
    fp32, int8 = result["models"]
    result["comparison"] = {
        "source_checkpoint_size_reduction_pct": 100.0 * (
            1.0 - int8["source_training_checkpoint_bytes"] / fp32["source_training_checkpoint_bytes"]
        ),
        "mean_latency_change_pct": 100.0 * (int8["latency"]["mean_ms"] / fp32["latency"]["mean_ms"] - 1.0),
        "peak_memory_change_pct": 100.0 * (int8["memory"]["peak_allocated_mib"] / fp32["memory"]["peak_allocated_mib"] - 1.0),
    }
    if args.fp32_deployment and args.int8_deployment:
        deploy_fp32, deploy_int8 = Path(args.fp32_deployment), Path(args.int8_deployment)
        result["deployment_artifacts"] = {
            "fp32": {"path": str(deploy_fp32), "bytes": deploy_fp32.stat().st_size, "size_mib": deploy_fp32.stat().st_size / 1024.0 / 1024.0},
            "int8": {"path": str(deploy_int8), "bytes": deploy_int8.stat().st_size, "size_mib": deploy_int8.stat().st_size / 1024.0 / 1024.0},
        }
        result["comparison"]["deployment_weight_size_reduction_pct"] = 100.0 * (
            1.0 - deploy_int8.stat().st_size / deploy_fp32.stat().st_size
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
