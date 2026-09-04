"""Import pipeline: pose extraction -> student inference -> skeleton video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.common import log_event
from dahua_cup.pipeline.render_rtmpose17_pose import render as render_pose_video
from dahua_cup.pipeline.rtmpose17_student_worker import (
    initialize_model,
    labels,
    load_feature,
    temperature_scale_probabilities,
)

PROTOGCN_DEPLOY_CONFIG = (
    "third_party/ProtoGCN/configs/campus6/rtm_s_coco17_k400_2d_gap_full.py"
)
STUDENT_CHECKPOINT = "models/student/M1KD.int8.pt"

_student_lock = threading.Lock()
_student: dict = {}


def student_model(config: Path, checkpoint: Path, device: str):
    """Build the M1KD INT8 model once and reuse it across imports."""
    with _student_lock:
        if not _student:
            model, loaded_format, metadata = initialize_model(
                config, checkpoint, device, "quantized")
            _student.update(model=model, format=loaded_format, metadata=metadata)
        return _student


def run_pose_worker(pose_python: str, video: Path, feature: Path, device: str,
                    max_frames: int = 100) -> None:
    """Extract COCO-17 poses with the RTMPose env via a subprocess."""
    repository_root = os.environ.get("DAHUA_CODE_ROOT", "")
    environment = dict(os.environ)
    if repository_root:
        environment["DAHUA_CODE_ROOT"] = repository_root
        environment["PYTHONPATH"] = str(Path(repository_root) / "src") + \
            os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        pose_python or sys.executable, "-m", "dahua_cup.pipeline.rtmpose17_pose_worker",
        "--video", str(video), "--feature", str(feature), "--device", device,
        "--frame-stride", "1", "--max-frames", str(max_frames),
        "--bbox-score", "0.15", "--joint-score-threshold", "0.20",
    ]
    subprocess.run(command, check=True, env=environment)


def infer_student(model, feature: Path, class_names: list[str], sample_id: str,
                  device: str, temperature: float = 1.0) -> dict:
    """Run ProtoGCN inference and return the calibrated top-6 prediction."""
    from protogcn.apis import inference_recognizer

    keypoint, score = load_feature(feature)
    video = {"keypoint": keypoint, "keypoint_score": score,
             "total_frames": keypoint.shape[1], "label": -1, "start_index": 0,
             "modality": "Pose", "test_mode": True}
    ranked = inference_recognizer(model, video)
    raw = np.zeros(len(class_names), dtype=np.float64)
    for index, raw_score in ranked:
        raw[int(index)] = float(raw_score)
    probabilities = temperature_scale_probabilities(raw, temperature)
    ordered = np.argsort(probabilities)[::-1]
    return {
        "schema_version": "protogcn_prediction.v1",
        "status": "completed",
        "sample_id": sample_id,
        "task": "campus6_rtmpose17",
        "label_space_size": 6,
        "modality": "joint",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "topk": [
            {"class_index": int(index), "label": class_names[int(index)],
             "score": float(probabilities[index])}
            for index in ordered[:6]
        ],
        "confidence_calibration": {
            "method": "temperature_scaling",
            "temperature": float(temperature),
            "top1_preserved": True,
        },
    }


def render_skeleton(feature: Path, output: Path, fps: float = 10.0) -> Path:
    """Render one skeleton stream video from a pose NPZ feature."""
    keypoint, score = load_feature(feature)
    source = output.with_suffix(".source.npz")
    np.savez_compressed(
        source, schema_version=np.asarray("rtmpose_coco17_2d.v1"),
        keypoint=keypoint, keypoint_score=score)
    try:
        render_pose_video(argparse.Namespace(
            feature=str(source), output=str(output), ffmpeg="ffmpeg",
            codec="libx264", preset="veryfast", bitrate="2M", fps=fps))
    finally:
        source.unlink(missing_ok=True)
    return output


def run_import_job(job: dict, dataset: "DemoDataset", settings: "DemoSettings") -> None:  # noqa: F821
    """Full pipeline for one uploaded video; updates the shared job dict."""
    sample_id = job["sample_id"]
    try:
        job["status"] = "running"
        dataset_root = dataset.root
        videos_dir = dataset_root / "videos" / "imported"
        derived = dataset_root / "derived" / sample_id
        videos_dir.mkdir(parents=True, exist_ok=True)
        derived.mkdir(parents=True, exist_ok=True)
        video = videos_dir / f"{sample_id}.mp4"
        shutil.move(str(job["source"]), video)
        job["source"] = video

        feature = derived / "feature.npz"
        run_pose_worker(settings.pose_python, video, feature, settings.device)
        log_event("demo_import_pose", sample_id=sample_id, video=str(video))

        model = student_model(settings.student_config, settings.student_checkpoint,
                              settings.device)["model"]
        class_names = labels(settings.label_map)
        prediction = infer_student(model, feature, class_names, sample_id,
                                   settings.device, settings.temperature)
        prediction_path = derived / "prediction.json"
        prediction_path.write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        pose_path = render_skeleton(feature, derived / "pose.mp4")

        row = {
            "id": sample_id,
            "label": prediction["topk"][0]["label"],
            "original_name": job.get("original_name", video.name),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "rgb": str(video.relative_to(dataset_root)),
            "pose": str(pose_path.relative_to(dataset_root)),
            "prediction": str(prediction_path.relative_to(dataset_root)),
            "source": "imported",
        }
        dataset.add_imported(row)
        job["status"] = "completed"
        job["sample"] = {"id": sample_id, "label": row["label"]}
        log_event("demo_import_complete", sample_id=sample_id, top1=row["label"])
    except Exception as error:  # surface any pipeline failure to the frontend
        job["status"] = "failed"
        job["message"] = f"{type(error).__name__}: {error}"
        log_event("demo_import_failed", sample_id=sample_id, error=str(error))
