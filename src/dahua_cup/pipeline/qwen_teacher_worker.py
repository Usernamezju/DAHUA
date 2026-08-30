"""Run one Qwen3-VL teacher inference from existing Web pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dahua_cup.feature_extraction.semantic_graph import (
    summarize_pose_feature as summarize_measured_pose_feature,
)
from dahua_cup.pipeline.common import file_hash, log_event, require_file
from dahua_cup.paths import CONFIG_ROOT
from dahua_cup.semantic_teacher.api.qwen3_backend import Qwen3Config, Qwen3Teacher

DEFAULT_LABEL_MAP = CONFIG_ROOT / "campus/campus6_labels.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--pose-video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student-json")
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--model-dir",
        default=str(
            Path(
                os.environ.get(
                    "DAHUA_MODEL_ROOT",
                    "/workspace/data/xzz_data/DAHUA/models",
                )
            )
            / "Qwen3-VL-32B-Instruct"
        ),
    )
    parser.add_argument("--revision")
    parser.add_argument(
        "--dtype",
        default=os.environ.get("DAHUA_QWEN_DTYPE", "float16"),
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("DAHUA_QWEN_ATTN_IMPLEMENTATION", "sdpa"),
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Attention backend; SDPA works without the optional flash-attn package",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    return parser


def load_labels(path: str | Path) -> tuple[str, ...]:
    labels = tuple(
        line.strip()
        for line in require_file(path, "teacher label map")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("teacher label map must contain unique non-empty labels")
    return labels


def load_student_hint(
    path: str | Path | None,
    allowed_labels: tuple[str, ...],
) -> tuple[str, dict[str, float]]:
    if not path:
        return "", {}
    source = Path(path)
    if not source.is_file():
        return "", {}
    value = json.loads(source.read_text(encoding="utf-8"))
    allowed = set(allowed_labels)
    distribution = {
        str(item["label"]): float(item["score"])
        for item in value.get("topk", [])
        if str(item.get("label", "")) in allowed
    }
    return str(value.get("task", "")), distribution


def summarize_pose_feature(sample_id: str, path: str | Path) -> dict:
    """Teacher entry point for the shared, measured Campus6 pose evidence."""
    return summarize_measured_pose_feature(sample_id, path)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_frames < 2 or args.max_new_tokens <= 0:
        raise ValueError(
            "max_frames must be at least 2 and max_new_tokens must be positive"
        )
    pose_video = require_file(args.pose_video, "rendered pose video")
    labels = load_labels(args.label_map)
    student_task, student_distribution = load_student_hint(
        args.student_json, labels
    )
    task = args.task.strip() or student_task or f"closed_set_{len(labels)}"
    semantic_graph = summarize_pose_feature(args.sample_id, args.feature)
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Qwen3-VL model directory not found: {model_dir}")

    teacher = Qwen3Teacher(
        Qwen3Config(
            model_dir=str(model_dir),
            revision=args.revision,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation or None,
            local_files_only=True,
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
            retries=args.retries,
        )
    )
    result, provenance = teacher.predict(
        sample_id=args.sample_id,
        semantic_graph=semantic_graph,
        student_distribution=student_distribution,
        video_path=pose_video,
        allowed_labels=labels,
        task=task,
    )
    output = {
        "schema_version": "teacher_prediction.v1",
        "status": "completed",
        "task": task,
        "label_space_size": len(labels),
        "model": provenance["teacher_model_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result.to_dict(),
        "provenance": {
            **provenance,
            "model_dir": str(model_dir),
            "torch_dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "label_map": str(Path(args.label_map).resolve()),
            "label_map_sha256": file_hash(args.label_map),
            "pose_video": str(pose_video),
            "pose_feature": str(Path(args.feature).resolve()),
            "max_frames": args.max_frames,
            "max_new_tokens": args.max_new_tokens,
        },
        "semantic_graph": semantic_graph,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    log_event(
        "qwen_teacher_complete",
        sample_id=args.sample_id,
        task=task,
        label=result.label,
        label_space_size=len(labels),
        output=str(destination),
    )


if __name__ == "__main__":
    main()
