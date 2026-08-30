"""Run the trained Campus6 ProtoGCN model on a direct RTMPose COCO-17 feature."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dahua_cup.feature_extraction.semantic_graph import summarize_ntu25_pose_feature
from dahua_cup.pipeline.common import file_hash, log_event, require_file
from dahua_cup.pipeline.protogcn_student_worker import previous_inference_history


ROOT = Path(__file__).resolve().parents[2]
PROTOGCN_ROOT = ROOT / "gcn_models" / "ProtoGCN"
DEFAULT_CONFIG = PROTOGCN_ROOT / "configs/campus6/rtmpose26_k400_2d_gap_full.py"
DEFAULT_LABELS = ROOT / "dahua_cup/configs/campus/campus6_labels.txt"
DEFAULT_CHECKPOINT = Path("/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/campus6_rtmpose26_k400_2d_gap_full_manual_v2/best_top1_acc_epoch_40.pth")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sample-id", required=True); value.add_argument("--feature", required=True)
    value.add_argument("--output", required=True); value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--checkpoint", default=os.environ.get("DAHUA_CAMPUS6_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    value.add_argument("--label-map", default=str(DEFAULT_LABELS)); value.add_argument("--device", default="cuda:0")
    value.add_argument("--topk", type=int, default=5)
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


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.topk <= 5: raise ValueError("topk must be in [1,5]")
    feature = require_file(args.feature, "RTMPose17 feature"); config = require_file(args.config, "Campus6 config")
    checkpoint = require_file(args.checkpoint, "Campus6 checkpoint"); class_names = labels(args.label_map)
    keypoint, score = load_feature(feature)
    sys.path.insert(0, str(PROTOGCN_ROOT))
    # Ensure the GAP recognizer registers even when an old package __init__ is cached.
    from protogcn.models.recognizers.recognizergcn_gap import RecognizerGCNGAP  # noqa: F401
    from protogcn.apis import inference_recognizer, init_recognizer
    model = init_recognizer(str(config), str(checkpoint), device=args.device)
    video = {"keypoint": keypoint, "keypoint_score": score, "total_frames": keypoint.shape[1],
             "label": -1, "start_index": 0, "modality": "Pose"}
    ranked = inference_recognizer(model, video)
    topk = [{"class_index": int(index), "label": class_names[int(index)], "score": float(score)}
            for index, score in ranked[: min(args.topk, len(ranked))]]
    output_path = Path(args.output)
    semantic = summarize_ntu25_pose_feature(args.sample_id, feature)
    output = {"schema_version": "protogcn_prediction.v1", "status": "completed", "sample_id": args.sample_id,
              "task": "campus6_rtmpose17", "label_space_size": 6, "modality": "joint",
              "generated_at": datetime.now(timezone.utc).isoformat(), "feature": str(feature), "config": str(config),
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint), "device": args.device,
              "physical_gpu_id": os.environ.get("DAHUA_PHYSICAL_GPU_ID", ""), "topk": topk,
              "student_evidence": {"schema_version": "protogcn_measured_evidence.v1", "semantic_graph": semantic},
              "inference_history": previous_inference_history(output_path), "warning": "Campus6 test accuracy is 47/53 (88.68%); conflict_chase test n=2."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(output_path)
    log_event("campus6_rtmpose17_inference_complete", output=str(output_path), top1=topk[0])


if __name__ == "__main__": main()
