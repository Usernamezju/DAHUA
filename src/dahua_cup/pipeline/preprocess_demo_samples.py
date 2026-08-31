"""Precompute demo assets for the selected 25+25 samples.

KTH rows get: avi -> browser-safe mp4, RTMPose COCO-17 features, M1KD student
predictions and skeleton stream videos.  Campus6 rows reuse the official
baseline keypoints, so only the skeleton stream video is rendered; their
predictions stay in ``campus6_baseline/M1KD.eval_all.pkl``.

Run on the server with the runtime environment that has mmpose + torch.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="ProtoGCN Campus6 deployment config")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="M1KD INT8 checkpoint")
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--pose-python", default=None,
                        help="python for RTMPose extraction (env with mmpose>=1.0); "
                             "defaults to the current interpreter")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--skip", default="", help="comma list of kth/campus6 to skip")
    return parser.parse_args()


def transcode_avi(source: Path, destination: Path) -> Path:
    """Convert KTH avi to a browser-safe h264 mp4 next to the source."""
    if destination.is_file() and destination.stat().st_size:
        return destination
    command = [
        "ffmpeg", "-y", "-i", str(source), "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-loglevel", "error", str(destination),
    ]
    subprocess.run(command, check=True)
    return destination


def load_selection(dataset: Path) -> dict:
    path = dataset / "selection.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_selection(dataset: Path, selection: dict) -> None:
    temporary = dataset / "selection.json.tmp"
    temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(dataset / "selection.json")


def run_pose_extraction(args, video: Path, feature: Path) -> None:
    """Extract COCO-17 poses with the RTMPose env, possibly a subprocess."""
    if feature.is_file() and feature.stat().st_size:
        return
    environment = dict(os.environ)
    repository_root = os.environ.get(
        "DAHUA_CODE_ROOT", str(Path(__file__).resolve().parents[3]))
    environment["DAHUA_CODE_ROOT"] = repository_root
    environment["PYTHONPATH"] = str(Path(repository_root) / "src") + os.pathsep + \
        environment.get("PYTHONPATH", "")
    command = [
        args.pose_python or sys.executable, "-m", "dahua_cup.pipeline.rtmpose17_pose_worker",
        "--video", str(video), "--feature", str(feature), "--device", args.device,
        "--frame-stride", "1", "--max-frames", str(args.max_frames),
        "--bbox-score", "0.15", "--joint-score-threshold", "0.20",
    ]
    subprocess.run(command, check=True, env=environment)


def run_kth_student(model, class_names, args, sample_id: str, feature: Path,
                    temperature: float) -> dict:
    keypoint, score = load_feature(feature)
    from protogcn.apis import inference_recognizer

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
        "device": args.device,
        "topk": [
            {"class_index": int(index), "label": class_names[int(index)],
             "score": float(probabilities[index])}
            for index in ordered[:6]
        ],
        "confidence_calibration": {
            "method": "temperature_scaling",
            "temperature": temperature,
            "top1_preserved": True,
        },
    }


def render_from_keypoints(dataset: Path, sample_id: str, keypoint: np.ndarray,
                          score: np.ndarray, fps: float = 10.0) -> Path:
    """Render one skeleton stream video from raw keypoints."""
    derived = dataset / "derived" / sample_id
    derived.mkdir(parents=True, exist_ok=True)
    feature = derived / "render_source.npz"
    output = derived / "pose.mp4"
    if output.is_file() and output.stat().st_size:
        return output
    np.savez_compressed(
        feature, schema_version=np.asarray("rtmpose_coco17_2d.v1"),
        keypoint=keypoint, keypoint_score=score)
    try:
        render_pose_video(argparse.Namespace(
            feature=str(feature), output=str(output), ffmpeg="ffmpeg",
            codec="libx264", preset="veryfast", bitrate="2M", fps=fps))
    finally:
        feature.unlink(missing_ok=True)
    return output


def main(argv=None):
    args = parse_args()
    dataset = args.dataset.resolve()
    selection = load_selection(dataset)
    class_names = labels(args.label_map)
    skip = {value.strip() for value in args.skip.split(",") if value.strip()}

    # KTH: transcode, pose extraction, student inference, skeleton video.
    if "kth" not in skip and selection.get("kth"):
        model, _, _ = initialize_model(
            args.config, args.checkpoint, args.device, "quantized")
        for row in selection["kth"]:
            sample_id = row["id"]
            source = dataset / row["rgb"]
            if source.suffix.lower() == ".avi":
                row["rgb"] = str(transcode_avi(
                    source, source.with_suffix(".mp4")).relative_to(dataset))
                source = dataset / row["rgb"]
            derived = dataset / "derived" / sample_id
            derived.mkdir(parents=True, exist_ok=True)
            feature = derived / "feature.npz"
            run_pose_extraction(args, source, feature)
            log_event("demo_pose_extract", sample_id=sample_id, video=str(source))
            prediction_path = derived / "prediction.json"
            prediction = run_kth_student(
                model, class_names, args, sample_id, feature, args.temperature)
            prediction_path.write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            row["prediction"] = str(prediction_path.relative_to(dataset))
            row["pose"] = str(
                render_from_keypoints(dataset, sample_id, *load_feature(feature),
                                      fps=10.0).relative_to(dataset))
            log_event("demo_kth_ready", sample_id=sample_id,
                      top1=prediction["topk"][0]["label"])
        del model
        save_selection(dataset, selection)

    # Campus6: skeleton videos from the official baseline keypoints.
    if "campus6" not in skip and selection.get("campus6"):
        annotations_path = dataset / "campus6_baseline" / "annotations_with_all.pkl"
        with annotations_path.open("rb") as stream:
            payload = pickle.load(stream)
        annotations = list(payload.get("annotations") or [])
        for row in selection["campus6"]:
            sample_id = row["id"]
            annotation = annotations[int(row["annotation_index"])]
            keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
            score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
            row["pose"] = str(
                render_from_keypoints(dataset, sample_id, keypoint, score, fps=10.0)
                .relative_to(dataset))
            log_event("demo_campus6_ready", sample_id=sample_id)
        save_selection(dataset, selection)

    print(json.dumps({"status": "complete", "dataset": str(dataset)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
