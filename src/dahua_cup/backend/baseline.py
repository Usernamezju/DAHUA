"""Read-only adapter for existing Campus6 skeletons, labels, and predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from dahua_cup.pipeline.common import file_hash
from dahua_cup.pipeline.render_rtmpose17_pose import render
from dahua_cup.feature_extraction.semantic_graph import summarize_pose_arrays
from dahua_cup.semantic_teacher.schemas import LABELS


class Campus6Baseline:
    """Index official server artefacts without extracting or inferring again."""

    def __init__(self, annotations: Path, predictions: Optional[Path] = None):
        self.annotations_path = Path(annotations).resolve()
        self.predictions_path = Path(predictions).resolve() if predictions else None
        self._lock = threading.Lock()
        self._annotations: dict[str, dict] = {}
        self._records: list[dict] = []
        self._predictions: dict[str, dict] = {}
        if self.annotations_path.is_file():
            self._load()

    @property
    def available(self) -> bool:
        return bool(self._records)

    @staticmethod
    def _sample_id(index: int, frame_dir: str) -> str:
        digest = hashlib.sha256(frame_dir.encode("utf-8")).hexdigest()[:8]
        return f"campus6_{index:04d}_{digest}"

    def _load(self) -> None:
        with self.annotations_path.open("rb") as stream:
            payload = pickle.load(stream)
        annotations = list(payload.get("annotations") or [])
        by_name = {str(item["frame_dir"]): item for item in annotations}
        order = list((payload.get("split") or {}).get("all") or by_name)
        split_for: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for name in (payload.get("split") or {}).get(split, []):
                split_for[str(name)] = split
        scores = None
        prediction_metadata = {}
        if self.predictions_path and self.predictions_path.is_file():
            with self.predictions_path.open("rb") as stream:
                scores = np.asarray(pickle.load(stream), dtype=np.float32)
            if scores.shape != (len(order), len(LABELS)):
                raise ValueError(
                    f"Campus6 prediction shape {scores.shape} does not match "
                    f"({len(order)}, {len(LABELS)})"
                )
            sidecar = self.predictions_path.with_suffix(
                self.predictions_path.suffix + ".json"
            )
            if sidecar.is_file():
                prediction_metadata = json.loads(
                    sidecar.read_text(encoding="utf-8")
                )
        score_index = {
            str(name): index for index, name in enumerate(order)
        }
        if prediction_metadata.get("order") == "annotations":
            score_index = {
                str(item["frame_dir"]): index
                for index, item in enumerate(annotations)
            }
        checkpoint = Path(str(prediction_metadata.get("checkpoint") or ""))
        checkpoint_hash = (
            file_hash(checkpoint) if checkpoint.is_file() else ""
        )
        generated = datetime.fromtimestamp(
            (self.predictions_path or self.annotations_path).stat().st_mtime,
            timezone.utc,
        ).isoformat()
        for index, frame_dir in enumerate(order):
            annotation = by_name[str(frame_dir)]
            identifier = self._sample_id(index, str(frame_dir))
            label = LABELS[int(annotation["label"])]
            self._annotations[identifier] = annotation
            self._records.append({
                "sample_id": identifier,
                "video_path": str(self.annotations_path),
                "source_dataset": "Campus6_initial",
                "source_label": label,
                "manual_label": label,
                "status": "reviewed",
                "reviewer": "official_initial_annotation",
                "reason_code": "official_ground_truth",
                "note": "split=" + split_for.get(str(frame_dir), "all"),
            })
            if scores is not None:
                values = scores[score_index[str(frame_dir)]]
                ranked = np.argsort(values)[::-1][:6]
                self._predictions[identifier] = {
                    "schema_version": "protogcn_prediction.v1",
                    "status": "completed",
                    "sample_id": identifier,
                    "task": "campus6_rtmpose17",
                    "label_space_size": len(LABELS),
                    "modality": "joint",
                    "generated_at": generated,
                    "checkpoint": "models/student/M1KD.int8.pt",
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_format": "portable_int8",
                    "model_role": "production student inference",
                    "evaluation_protocol": prediction_metadata.get(
                        "evaluation_protocol", ""
                    ),
                    "evaluation_summary": {
                        key: prediction_metadata.get(key)
                        for key in ("correct", "total", "accuracy")
                        if key in prediction_metadata
                    },
                    "confidence_calibration": prediction_metadata.get(
                        "confidence_calibration", {}
                    ),
                    "topk": [
                        {
                            "class_index": int(class_index),
                            "label": LABELS[int(class_index)],
                            "score": float(values[class_index]),
                        }
                        for class_index in ranked
                    ],
                }
        counts = Counter(
            value["topk"][0]["label"]
            for value in self._predictions.values()
            if value.get("topk")
        )
        ordered = sorted(counts.values())
        cutoff = ordered[int((len(ordered) - 1) * 0.25)] if ordered else 0
        maximum = max(ordered, default=0)
        for value in self._predictions.values():
            if not value.get("topk"):
                continue
            label = value["topk"][0]["label"]
            value["predicted_class_count"] = counts[label]
            value["predicted_class_frequency"] = counts[label] / max(
                1, len(self._predictions)
            )
            value["rare_class"] = counts[label] <= cutoff and counts[label] < maximum

    def records(self) -> list[dict]:
        return [dict(item) for item in self._records]

    def has(self, sample_id: str) -> bool:
        return sample_id in self._annotations

    def sample_ids(self) -> list[str]:
        return list(self._annotations)

    def prediction(self, sample_id: str) -> Optional[dict]:
        value = self._predictions.get(sample_id)
        return dict(value) if value else None

    def evaluation_summary(self) -> dict:
        """Evaluate the indexed deployment outputs against official labels."""
        per_class = {
            label: {"correct": 0, "total": 0, "accuracy": 0.0}
            for label in LABELS
        }
        correct = total = 0
        for sample_id, annotation in self._annotations.items():
            prediction = self._predictions.get(sample_id) or {}
            topk = prediction.get("topk") or []
            if not topk:
                continue
            expected = LABELS[int(annotation["label"])]
            predicted = str(topk[0]["label"])
            matched = predicted == expected
            total += 1
            correct += int(matched)
            per_class[expected]["total"] += 1
            per_class[expected]["correct"] += int(matched)
        for value in per_class.values():
            value["accuracy"] = (
                value["correct"] / value["total"] if value["total"] else 0.0
            )
        populated = [value["accuracy"] for value in per_class.values() if value["total"]]
        return {
            "overall_accuracy": correct / total if total else 0.0,
            "mean_class_accuracy": (
                sum(populated) / len(populated) if populated else 0.0
            ),
            "correct": correct,
            "total": total,
            "per_class": per_class,
        }

    def pose_metrics(self, sample_id: str) -> dict:
        annotation = self._annotations[sample_id]
        score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
        valid = score >= 0.20
        active_people = valid.any(axis=(1, 2))
        if not active_people.any():
            return {
                "quality": 0.0,
                "pose_coverage": 0.0,
                "mean_joint_score": 0.0,
                "active_people": 0,
                "status": "no_pose",
            }
        active_valid = valid[active_people]
        active_scores = score[active_people]
        coverage = float(active_valid.mean())
        mean_score = float(active_scores[active_valid].mean())
        return {
            "quality": max(0.0, min(1.0, 0.70 * coverage + 0.30 * mean_score)),
            "pose_coverage": coverage,
            "mean_joint_score": mean_score,
            "active_people": int(active_people.sum()),
            "status": "ok",
        }

    def semantic_graph(self, sample_id: str) -> dict:
        annotation = self._annotations[sample_id]
        keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
        score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
        return summarize_pose_arrays(
            sample_id, keypoint, score >= 0.20, fps=10.0
        )

    def materialize_feature(self, sample_id: str, destination: Path) -> Path:
        """Expose an existing skeleton as the normal worker NPZ contract."""
        if destination.is_file() and destination.stat().st_size:
            return destination
        annotation = self._annotations[sample_id]
        keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
        score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray("rtmpose_coco17_2d.v1"),
                keypoint=keypoint,
                keypoint_score=score,
                valid_mask=score >= 0.20,
                fps=np.asarray(10.0, dtype=np.float32),
            )
        temporary.replace(destination)
        return destination

    def render_pose_video(self, sample_id: str, destination: Path) -> Path:
        """Render, but never re-extract, an existing skeleton sequence."""
        if destination.is_file() and destination.stat().st_size:
            return destination
        annotation = self._annotations[sample_id]
        with self._lock:
            if destination.is_file() and destination.stat().st_size:
                return destination
            feature = destination.with_suffix(".source.npz")
            keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
            score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature,
                schema_version=np.asarray("rtmpose_coco17_2d.v1"),
                keypoint=keypoint,
                keypoint_score=score,
            )
            args = argparse.Namespace(
                feature=str(feature), output=str(destination), ffmpeg="ffmpeg",
                codec="libx264", preset="veryfast", bitrate="2M", fps=10.0,
            )
            try:
                render(args)
            finally:
                feature.unlink(missing_ok=True)
        return destination
