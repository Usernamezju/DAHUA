"""Classify one NTU-25 pose artifact with an official ProtoGCN checkpoint."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GCN_MODELS_ROOT = REPOSITORY_ROOT / "gcn_models"
PROTOGCN_ROOT = GCN_MODELS_ROOT / "ProtoGCN"
DEFAULT_CONFIG = PROTOGCN_ROOT / "configs/ntu120_xsub/b.py"
DEFAULT_LABEL_MAP = GCN_MODELS_ROOT / "GAP/text/ntu120_label_map.txt"
EXPECTED_FEATURE_SCHEMA = "mediapipe_ntu25.v1"
MAX_INFERENCE_HISTORY = 4


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DAHUA_PROTOGCN_PRETRAINED"),
        help="ProtoGCN checkpoint (or DAHUA_PROTOGCN_PRETRAINED)",
    )
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=5)
    return parser


def load_labels(path):
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"label map is empty: {path}")
    return labels


def load_pose_feature(path):
    with np.load(path, allow_pickle=False) as artifact:
        if "schema_version" not in artifact or "keypoint" not in artifact:
            raise ValueError(f"not a {EXPECTED_FEATURE_SCHEMA} artifact: {path}")
        schema = str(artifact["schema_version"].item())
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)

    if schema != EXPECTED_FEATURE_SCHEMA:
        raise ValueError(f"expected schema {EXPECTED_FEATURE_SCHEMA}, got {schema}")
    if keypoint.ndim != 4 or keypoint.shape[0] not in (1, 2) or keypoint.shape[2:] != (25, 3):
        raise ValueError(f"keypoint must have shape (1|2, T, 25, 3), got {keypoint.shape}")
    if keypoint.shape[1] == 0:
        raise ValueError("pose artifact contains zero frames")
    if not np.isfinite(keypoint).all():
        raise ValueError("pose artifact contains non-finite coordinates")
    if np.allclose(keypoint, 0.0):
        raise ValueError("MediaPipe found no usable pose in the video")
    return keypoint


def prediction_summary(value: dict) -> dict | None:
    topk = value.get("topk") or []
    if not topk:
        return None
    try:
        top1_score = float(topk[0]["score"])
        top2_score = float(topk[1]["score"]) if len(topk) > 1 else None
        return {
            "generated_at": value.get("generated_at", ""),
            "top1_label": str(topk[0]["label"]),
            "top1_score": top1_score,
            "top1_top2_margin": (
                max(0.0, top1_score - top2_score)
                if top2_score is not None else None
            ),
        }
    except (KeyError, TypeError, ValueError):
        return None


def previous_inference_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    history = list(previous.get("inference_history") or [])
    summary = prediction_summary(previous)
    if summary is not None:
        history.append(summary)
    return history[-MAX_INFERENCE_HISTORY:]


def build_student_evidence(
    sample_id: str, feature_path: str | Path, topk: list[dict]
) -> dict:
    """Create an auditable explanation from the actual ProtoGCN inputs and scores.

    This deliberately reports measurements and decision statistics, not an
    unsupported natural-language claim about why a particular action occurred.
    """
    graph = summarize_ntu25_pose_feature(sample_id, feature_path)
    persons = list(graph["persons"])
    active = [item for item in persons if item["frame_coverage"] > 0.0]
    top1 = topk[0] if topk else {}
    top2 = topk[1] if len(topk) > 1 else {}
    top1_score = float(top1.get("score", 0.0))
    top2_score = float(top2.get("score", 0.0))
    margin = max(0.0, top1_score - top2_score)
    coverage = float(graph["quality"]["pose_coverage"])
    duration_ms = int(graph["segments"][0]["end_ms"])
    limitations = []
    if coverage < 0.70:
        limitations.append(
            {
                "code": "low_pose_coverage",
                "description": "骨架关节点覆盖率低于 70%，需结合原始视频人工复核。",
            }
        )
    if len(active) < 2:
        limitations.append(
            {
                "code": "limited_interaction_evidence",
                "description": "有效骨架轨迹少于两人，不能据此推断双人交互关系。",
            }
        )
    evidence = [
        {
            "type": "model_decision",
            "description": (
                f"Top-1 为 {top1.get('label', 'unknown')}（{top1_score:.1%}），"
                f"与 Top-2 的概率差为 {margin:.1%}。"
            ),
            "top1_top2_margin": margin,
            "top5_probability_mass": float(sum(float(item["score"]) for item in topk)),
        },
        {
            "type": "pose_coverage",
            "description": (
                f"片段时长 {duration_ms / 1000.0:.2f} 秒；检测到 {len(active)} 条有效人体轨迹，"
                f"关节点覆盖率 {coverage:.1%}。"
            ),
            "duration_ms": duration_ms,
            "active_person_count": len(active),
            "pose_coverage": coverage,
        },
    ]
    for person in active:
        evidence.append(
            {
                "type": "person_motion",
                "description": (
                    f"轨迹 {person['id']}：帧覆盖率 {person['frame_coverage']:.1%}，"
                    f"平均归一化重心速度 {person['mean_normalized_speed']:.3f}/秒。"
                ),
                **person,
            }
        )
    for relation in graph["relations"]:
        evidence.append(
            {
                "type": relation["type"],
                "description": (
                    f"轨迹 {relation['src']} 与 {relation['dst']} 同时可见帧占比 "
                    f"{relation['frame_coverage']:.1%}；平均归一化距离 "
                    f"{relation['mean_normalized_distance']:.3f}。"
                ),
                **relation,
            }
        )
    return {
        "schema_version": "protogcn_measured_evidence.v1",
        "method": "semantic_graph.measured_pose_and_student_margin",
        "semantic_graph": graph,
        "evidence": evidence,
        "limitations": limitations,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    feature_path = require_file(args.feature, "pose feature")
    config_path = require_file(args.config, "ProtoGCN config")
    if not args.checkpoint:
        raise ValueError("--checkpoint or DAHUA_PROTOGCN_PRETRAINED is required")
    checkpoint_path = require_file(args.checkpoint, "ProtoGCN checkpoint")
    label_map_path = require_file(args.label_map, "NTU120 label map")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.topk > 5:
        raise ValueError("ProtoGCN's inference API returns at most five classes")

    keypoint = load_pose_feature(feature_path)
    labels = load_labels(label_map_path)
    destination = Path(args.output)

    sys.path.insert(0, str(PROTOGCN_ROOT))
    try:
        from protogcn.apis import inference_recognizer, init_recognizer
    except ImportError as exc:
        raise RuntimeError(
            "ProtoGCN runtime dependencies are unavailable in this Python environment"
        ) from exc

    model = init_recognizer(str(config_path), str(checkpoint_path), device=args.device)
    video = {
        "keypoint": keypoint,
        "total_frames": keypoint.shape[1],
        "label": -1,
        "start_index": 0,
        "modality": "Pose",
    }
    ranked = inference_recognizer(model, video)
    topk = []
    for class_index, score in ranked[: min(args.topk, len(ranked))]:
        index = int(class_index)
        topk.append(
            {
                "class_index": index,
                "label": labels[index] if index < len(labels) else f"class_{index}",
                "score": float(score),
            }
        )

    student_evidence = build_student_evidence(
        args.sample_id or destination.stem, feature_path, topk
    )

    output = {
        "schema_version": "protogcn_prediction.v1",
        "sample_id": args.sample_id or destination.stem,
        "task": "ntu120_xsub",
        "label_space_size": len(labels),
        "modality": "bone",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature": str(feature_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "device": args.device,
        "physical_gpu_id": os.environ.get("DAHUA_PHYSICAL_GPU_ID", ""),
        "topk": topk,
        "student_evidence": student_evidence,
        "inference_history": previous_inference_history(destination),
        "warning": (
            "MediaPipe world landmarks differ from Kinect NTU skeletons; "
            "this result validates the pipeline, not campus-domain accuracy."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    log_event("protogcn_inference_complete", output=str(destination), top1=topk[0])


if __name__ == "__main__":
    main()
