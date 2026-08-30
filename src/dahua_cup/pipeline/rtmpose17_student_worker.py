"""Run the trained Campus6 ProtoGCN model on a direct RTMPose COCO-17 feature."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dahua_cup.feature_extraction.semantic_graph import summarize_pose_feature
from dahua_cup.pipeline.common import file_hash, log_event, require_file
from dahua_cup.paths import CONFIG_ROOT, PROTOGCN_ROOT

DEFAULT_CONFIG = PROTOGCN_ROOT / "configs/campus6/rtmpose26_k400_2d_gap_full.py"
DEFAULT_LABELS = CONFIG_ROOT / "campus/campus6_labels.txt"
DEFAULT_CHECKPOINT = Path("/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/m1kd_best_full/M1KD.runtime.int8.pt")
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
    value.add_argument("--checkpoint-format", choices=("auto", "fp32", "quantized"), default="auto")
    value.add_argument("--label-map", default=str(DEFAULT_LABELS)); value.add_argument("--device", default="cuda:0")
    value.add_argument("--topk", type=int, default=6)
    value.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="validation-fitted probability temperature; 1.0 disables calibration",
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


def initialize_model(config: Path, checkpoint: Path, device: str, requested_format: str):
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
    from protogcn.apis import init_recognizer
    return init_recognizer(str(config), str(checkpoint), device=device), kind, {}


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.topk <= 6: raise ValueError("topk must be in [1,6]")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be a positive finite number")
    feature = require_file(args.feature, "RTMPose17 feature"); config = require_file(args.config, "Campus6 config")
    checkpoint = require_file(args.checkpoint, "Campus6 checkpoint"); class_names = labels(args.label_map)
    keypoint, score = load_feature(feature)
    sys.path.insert(0, str(PROTOGCN_ROOT))
    from protogcn.apis import inference_recognizer
    model, loaded_format, deployment_metadata = initialize_model(
        config, checkpoint, args.device, args.checkpoint_format
    )
    video = {"keypoint": keypoint, "keypoint_score": score, "total_frames": keypoint.shape[1],
             "label": -1, "start_index": 0, "modality": "Pose", "test_mode": True}
    ranked = inference_recognizer(model, video)
    raw_probabilities = np.zeros(len(class_names), dtype=np.float64)
    for index, raw_score in ranked:
        if not 0 <= int(index) < len(class_names):
            raise ValueError("ProtoGCN returned an invalid Campus6 class index")
        raw_probabilities[int(index)] = float(raw_score)
    probabilities = temperature_scale_probabilities(raw_probabilities, args.temperature)
    ordered = np.argsort(probabilities)[::-1]
    topk = [{"class_index": int(index), "label": class_names[int(index)], "score": float(probabilities[index])}
            for index in ordered[: min(args.topk, len(ordered))]]
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
    log_event("campus6_rtmpose17_inference_complete", output=str(output_path), top1=topk[0])


if __name__ == "__main__": main()
