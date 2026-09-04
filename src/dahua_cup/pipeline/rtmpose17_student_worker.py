"""Run the trained Campus6 ProtoGCN model on a direct RTMPose COCO-17 feature."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dahua_cup.feature_extraction.semantic_graph import summarize_pose_feature
from dahua_cup.pipeline.common import file_hash, log_event, require_file


def _first_existing(candidates: list[Path], description: str) -> Path:
    """Return a deployment resource without assuming one checkout layout.

    The packaged Web application keeps ProtoGCN in ``third_party/``.  The
    competition server's older checkout keeps it in ``gcn_models/`` instead.
    The worker must support both so the portable Huffman package does not
    silently depend on the Web application's source-tree layout.
    """
    for candidate in candidates:
        if candidate.is_dir() or candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found; checked: {rendered}")


def _runtime_paths() -> tuple[Path, Path]:
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parent
    try:
        from dahua_cup.paths import CONFIG_ROOT, PROTOGCN_ROOT

        configured_config_root = CONFIG_ROOT
        configured_protogcn_root = PROTOGCN_ROOT
    except ImportError:
        # Compatibility with the pre-src server checkout, which has no
        # ``dahua_cup.paths`` module.
        configured_config_root = package_root / "configs"
        configured_protogcn_root = repository_root / "third_party" / "ProtoGCN"
    configured = os.environ.get("DAHUA_PROTOGCN_ROOT", "").strip()
    protogcn_candidates = []
    if configured:
        protogcn_candidates.append(Path(configured).expanduser())
    protogcn_candidates.extend(
        [
            configured_protogcn_root,
            repository_root / "third_party" / "ProtoGCN",
            repository_root / "gcn_models" / "ProtoGCN",
            Path("/workspace/data/xzz_data/DAHUA/datasets/NTU/ProtoGCN"),
        ]
    )
    protogcn_root = _first_existing(
        protogcn_candidates,
        "ProtoGCN root",
    )
    config_root = _first_existing(
        [configured_config_root, package_root / "configs"], "Campus6 config root"
    )
    return protogcn_root, config_root


PROTOGCN_ROOT, CONFIG_ROOT = _runtime_paths()
DEFAULT_CONFIG = PROTOGCN_ROOT / "configs/campus6/rtm_s_coco17_k400_2d_gap_full.py"
DEFAULT_LABELS = CONFIG_ROOT / "campus/campus6_labels.txt"
DEFAULT_CHECKPOINT = Path(
    "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
    "m1kd_best_full/M1KD.runtime.int8.pt"
)
MAX_INFERENCE_HISTORY = 20


def previous_inference_history(path: Path) -> list[dict]:
    """Preserve a compact history when a sample is reclassified."""
    if not path.is_file():
        return []
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        history = list(previous.get("inference_history") or [])
        topk = list(previous.get("topk") or [])
        if topk:
            history.append({
                "generated_at": previous.get("generated_at", ""),
                "checkpoint": previous.get("checkpoint", ""),
                "top1": topk[0],
            })
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return history[-MAX_INFERENCE_HISTORY:]


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sample-id", required=True); value.add_argument("--feature", required=True)
    value.add_argument("--output", required=True); value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--checkpoint", default=os.environ.get("DAHUA_CAMPUS6_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    value.add_argument("--checkpoint-format", choices=("auto", "fp32", "quantized", "huffman"), default="auto")
    value.add_argument(
        "--residual-checkpoint",
        default="",
        help="state sidecar required by a Deep Compression Huffman checkpoint",
    )
    value.add_argument("--label-map", default=str(DEFAULT_LABELS)); value.add_argument("--device", default="cuda:0")
    value.add_argument("--topk", type=int, default=6)
    value.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="validation-fitted probability temperature; 1.0 disables calibration",
    )
    value.add_argument(
        "--ground-truth",
        default="",
        help="optional true Campus6 label; records per-sample accuracy in the timing log",
    )
    return value


def load_feature(path):
    with np.load(path, allow_pickle=False) as artifact:
        schema = str(np.asarray(artifact["schema_version"]).item())
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)
        score = np.asarray(artifact["keypoint_score"], dtype=np.float32) if "keypoint_score" in artifact else None
    if schema != "rtmpose_coco17_2d.v1" or keypoint.ndim != 4 or keypoint.shape[0] not in (1, 2) or keypoint.shape[2:] != (17, 2):
        raise ValueError(f"expected rtmpose_coco17_2d.v1 [M,T,17,2], got {schema} {keypoint.shape}")
    if score is not None and score.shape != keypoint.shape[:3]:
        raise ValueError("keypoint_score has incompatible shape")
    if keypoint.shape[1] < 1 or not np.isfinite(keypoint).all():
        raise ValueError("pose feature has no finite frames")
    if score is not None and not np.isfinite(score).all():
        raise ValueError("pose feature has non-finite scores")
    return keypoint, score


def labels(path):
    values = [line.strip() for line in require_file(path, "Campus6 label map").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 6 or len(set(values)) != 6:
        raise ValueError("Campus6 label map must contain six unique labels")
    return values


def checkpoint_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.read_bytes()[:5] == b"DCMP1":
        return "huffman"
    import torch

    try:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"unable to inspect Campus6 checkpoint: {exc}") from exc
    return "quantized" if isinstance(payload, dict) and "qparams" in payload else "fp32"


def temperature_scale_probabilities(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Calibrate a normalized class distribution without changing its argmax.

    ``inference_recognizer`` returns ProtoGCN's post-softmax, multi-clip
    probabilities. Applying softmax(log(p) / T) is therefore the
    probability-space form of standard temperature scaling.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Campus6 probabilities must be a non-empty vector")
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Campus6 probabilities must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Campus6 probability temperature must be positive")
    normalized = values / values.sum()
    adjusted = np.maximum(normalized, 1e-12) ** (1.0 / float(temperature))
    return (adjusted / adjusted.sum()).astype(np.float32)


def retarget_temperature_probabilities(
    probabilities: np.ndarray,
    source_temperature: float,
    target_temperature: float,
) -> np.ndarray:
    """Convert probabilities calibrated at one temperature to another.

    For a distribution generated as ``softmax(logits / source_temperature)``,
    applying a further temperature of ``target/source`` produces exactly
    ``softmax(logits / target_temperature)``.  This lets the review UI soften
    precomputed probability caches without changing their class order.
    """
    if not np.isfinite(source_temperature) or source_temperature <= 0:
        raise ValueError("Campus6 source probability temperature must be positive")
    if not np.isfinite(target_temperature) or target_temperature <= 0:
        raise ValueError("Campus6 target probability temperature must be positive")
    return temperature_scale_probabilities(
        probabilities, float(target_temperature) / float(source_temperature)
    )


def initialize_model(
    config: Path,
    checkpoint: Path,
    device: str,
    requested_format: str,
    residual_checkpoint: Path | None = None,
):
    """Build the correct inference graph and load FP32 or portable INT8 state."""
    sys.path.insert(0, str(PROTOGCN_ROOT))
    from protogcn.models.recognizers.recognizergcn_gap import RecognizerGCNGAP  # noqa: F401
    from protogcn.apis import init_recognizer

    kind = checkpoint_format(checkpoint, requested_format)
    if kind == "quantized":
        from dahua_cup.semantic_teacher.distillation.logits_kd import (
            load_quantized_state_dict,
        )

        model = init_recognizer(str(config), None, device=device)
        metadata = load_quantized_state_dict(model, checkpoint)
        model.eval()
        return model, kind, metadata
    if kind == "huffman":
        if residual_checkpoint is None:
            raise ValueError("a Huffman Deep Compression checkpoint requires --residual-checkpoint")
        import torch
        from dahua_cup.semantic_teacher.distillation.deep_compression import (
            campus6_training_only_state_keys,
            huffman_weight_keys,
            load_huffman_weights,
        )

        payload = torch.load(residual_checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError("invalid Deep Compression residual checkpoint")
        model = init_recognizer(str(config), None, device=device)
        incompatibility = model.load_state_dict(payload["state_dict"], strict=False)
        metadata = dict(payload.get("meta") or {})
        excluded = {str(name) for name in metadata.get("excluded_training_only_keys", [])}
        allowed_excluded = campus6_training_only_state_keys(model)
        unknown_excluded = excluded.difference(allowed_excluded)
        if unknown_excluded:
            raise ValueError(
                "Deep Compression residual declares non-training state as omitted: "
                f"{sorted(unknown_excluded)[:3]}"
            )
        expected_missing = huffman_weight_keys(checkpoint).union(excluded)
        if set(incompatibility.missing_keys) != expected_missing or incompatibility.unexpected_keys:
            raise ValueError(
                "Deep Compression residual does not match model/Huffman package: "
                f"missing={len(incompatibility.missing_keys)} unexpected={len(incompatibility.unexpected_keys)}"
            )
        metadata = load_huffman_weights(model, checkpoint)
        metadata["residual_checkpoint"] = str(residual_checkpoint)
        metadata["excluded_training_only_keys"] = len(excluded)
        model.eval()
        return model, kind, metadata
    from protogcn.apis import init_recognizer
    return init_recognizer(str(config), str(checkpoint), device=device), kind, {}


def main(argv=None):
    total_started = time.perf_counter()
    args = parser().parse_args(argv)
    if not 1 <= args.topk <= 6: raise ValueError("topk must be in [1,6]")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be a positive finite number")
    feature = require_file(args.feature, "RTMPose17 feature"); config = require_file(args.config, "Campus6 config")
    checkpoint = require_file(args.checkpoint, "Campus6 checkpoint"); class_names = labels(args.label_map)
    ground_truth = args.ground_truth.strip()
    if ground_truth and ground_truth not in class_names:
        raise ValueError("--ground-truth must be one of the six Campus6 labels")
    residual_checkpoint = (
        require_file(args.residual_checkpoint, "Deep Compression residual checkpoint")
        if args.residual_checkpoint
        else None
    )
    keypoint, score = load_feature(feature)
    sys.path.insert(0, str(PROTOGCN_ROOT))
    from protogcn.apis import inference_recognizer
    model_load_started = time.perf_counter()
    model, loaded_format, deployment_metadata = initialize_model(
        config, checkpoint, args.device, args.checkpoint_format, residual_checkpoint
    )
    model_load_seconds = time.perf_counter() - model_load_started
    video = {"keypoint": keypoint, "keypoint_score": score, "total_frames": keypoint.shape[1],
             "label": -1, "start_index": 0, "modality": "Pose", "test_mode": True}
    inference_started = time.perf_counter()
    ranked = inference_recognizer(model, video)
    inference_seconds = time.perf_counter() - inference_started
    raw_probabilities = np.zeros(len(class_names), dtype=np.float64)
    for index, raw_score in ranked:
        if not 0 <= int(index) < len(class_names):
            raise ValueError("ProtoGCN returned an invalid Campus6 class index")
        raw_probabilities[int(index)] = float(raw_score)
    probabilities = temperature_scale_probabilities(raw_probabilities, args.temperature)
    ordered = np.argsort(probabilities)[::-1]
    topk = [{"class_index": int(index), "label": class_names[int(index)], "score": float(probabilities[index])}
            for index in ordered[: min(args.topk, len(ordered))]]
    total_seconds = time.perf_counter() - total_started
    top1_label = topk[0]["label"]
    correct = ground_truth == top1_label if ground_truth else None
    output_path = Path(args.output)
    semantic = summarize_pose_feature(args.sample_id, feature)
    output = {"schema_version": "protogcn_prediction.v1", "status": "completed", "sample_id": args.sample_id,
              "task": "campus6_rtmpose17", "label_space_size": 6, "modality": "joint",
              "generated_at": datetime.now(timezone.utc).isoformat(), "feature": str(feature), "config": str(config),
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint), "checkpoint_format": loaded_format,
              "deployment_metadata": deployment_metadata, "device": args.device,
              "physical_gpu_id": os.environ.get("DAHUA_PHYSICAL_GPU_ID", ""), "topk": topk,
              "confidence_calibration": {"method": "temperature_scaling", "temperature": float(args.temperature), "top1_preserved": True},
              "student_evidence": {"schema_version": "protogcn_measured_evidence.v1", "semantic_graph": semantic},
              "timing": {
                  "model_load_seconds": round(model_load_seconds, 4),
                  "inference_seconds": round(inference_seconds, 4),
                  "total_seconds": round(total_seconds, 4),
              },
              "accuracy_log": {
                  "ground_truth": ground_truth,
                  "top1_label": top1_label,
                  "correct": correct,
              },
              "inference_history": previous_inference_history(output_path),
              "acceptance_metrics": {
                  "validation_accuracy": 47 / 53,
                  "test_accuracy": 46 / 53,
                  "all_accuracy": 345 / 358,
              },
              "warning": "Campus6 test split is small (n=53); conflict_chase test n=2."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(output_path)
    log_event("campus6_rtmpose17_inference_complete", output=str(output_path), top1=topk[0],
              model_load_seconds=round(model_load_seconds, 4), inference_seconds=round(inference_seconds, 4),
              total_seconds=round(total_seconds, 4), ground_truth=ground_truth, correct=correct)


if __name__ == "__main__": main()
