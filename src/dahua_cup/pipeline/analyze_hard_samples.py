"""Analyze the highest-value existing Campus6 hard samples with one Qwen load."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dahua_cup.backend.baseline import Campus6Baseline
from dahua_cup.backend.hard_samples import evaluate_hard_sample
from dahua_cup.semantic_teacher.api.qwen3_backend import Qwen3Config, Qwen3Teacher
from dahua_cup.semantic_teacher.schemas import LABELS


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", required=True)
    value.add_argument("--predictions", required=True)
    value.add_argument("--artifact-root", required=True)
    value.add_argument("--model-dir", required=True)
    value.add_argument("--confidence-threshold", type=float, default=0.30)
    value.add_argument("--margin-threshold", type=float, default=0.15)
    value.add_argument("--conflict-confidence-threshold", type=float, default=0.70)
    value.add_argument("--instability-threshold", type=float, default=0.60)
    value.add_argument("--limit", type=int, default=6)
    value.add_argument("--max-frames", type=int, default=8)
    value.add_argument("--max-new-tokens", type=int, default=512)
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.limit < 1:
        raise ValueError("limit must be positive")
    baseline = Campus6Baseline(Path(args.annotations), Path(args.predictions))
    rows = []
    for sample_id in baseline.sample_ids():
        prediction = baseline.prediction(sample_id)
        if prediction is None:
            continue
        decision = evaluate_hard_sample(
            prediction,
            None,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            conflict_confidence_threshold=args.conflict_confidence_threshold,
            instability_threshold=args.instability_threshold,
        )
        if decision["is_hard"]:
            rows.append({**prediction, "hard_decision": decision})
    rows.sort(key=lambda row: (
        -row["hard_decision"]["matched_condition_count"],
        "C1" not in row["hard_decision"]["matched_conditions"],
        row["sample_id"],
    ))
    selected = rows[:args.limit]
    root = Path(args.artifact_root).resolve()
    teacher = Qwen3Teacher(Qwen3Config(
        model_dir=str(Path(args.model_dir).resolve()),
        dtype="float16",
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        retries=2,
    ))
    completed = skipped = failed = 0
    for index, row in enumerate(selected, 1):
        sample_id = str(row["sample_id"])
        feature = root / "features" / f"{sample_id}.npz"
        pose_video = root / "pose_videos" / f"{sample_id}.mp4"
        prediction_path = root / "predictions" / f"{sample_id}.json"
        output = root / "teachers" / f"{sample_id}.json"
        if output.is_file() and output.stat().st_size:
            skipped += 1
            continue
        baseline.materialize_feature(sample_id, feature)
        baseline.render_pose_video(sample_id, pose_video)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        student_distribution = {
            item["label"]: float(item["score"])
            for item in row["topk"]
        }
        graph = baseline.semantic_graph(sample_id)
        try:
            result, provenance = teacher.predict(
                sample_id=sample_id,
                semantic_graph=graph,
                student_distribution=student_distribution,
                video_path=pose_video,
                allowed_labels=LABELS,
                task="campus6_hard_sample_analysis",
            )
        except Exception as exc:
            failure = {
                "schema_version": "teacher_prediction.v1",
                "status": "failed",
                "task": "campus6_hard_sample_analysis",
                "model": "Qwen3-VL-8B-Instruct",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "reason": f"{type(exc).__name__}: {exc}",
                "result": None,
                "semantic_graph": graph,
                "hard_sample": {
                    "rank": index,
                    **row["hard_decision"],
                },
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            failed += 1
            print(json.dumps({
                "processed": index,
                "total": len(selected),
                "sample_id": sample_id,
                "status": "failed",
                "reason": failure["reason"],
            }, ensure_ascii=False), flush=True)
            continue
        payload = {
            "schema_version": "teacher_prediction.v1",
            "status": "completed",
            "task": "campus6_hard_sample_analysis",
            "label_space_size": 6,
            "model": provenance["teacher_model_version"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
            "provenance": provenance,
            "semantic_graph": graph,
            "hard_sample": {
                "rank": index,
                **row["hard_decision"],
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        completed += 1
        print(json.dumps({
            "processed": index,
            "total": len(selected),
            "sample_id": sample_id,
            "teacher_label": result.label,
        }, ensure_ascii=False), flush=True)
    print(json.dumps({
        "status": "completed",
        "selected": len(selected),
        "analyzed": completed,
        "existing": skipped,
        "failed": failed,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
