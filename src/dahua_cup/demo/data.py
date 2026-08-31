"""Demo dataset access: selection index, predictions, and media paths."""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from dahua_cup.pipeline.rtmpose17_student_worker import temperature_scale_probabilities
from dahua_cup.semantic_teacher.schemas import LABELS

SELECTION_SCHEMA = "demo_selection.v1"


class DemoDataset:
    """Read ``selection.json`` and serve predictions for the demo samples."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        selection_path = self.root / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"selection.json not found under {self.root}")
        self.selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if self.selection.get("schema_version") != SELECTION_SCHEMA:
            raise ValueError("unsupported selection.json schema")
        self._rows: dict[str, dict] = {}
        for key in ("campus6", "kth", "imported"):
            for row in self.selection.get(key) or []:
                self._rows[str(row["id"])] = {**row, "group": key}
        self._lock = threading.Lock()
        self._eval_scores: Optional[np.ndarray] = None
        self._eval_temperature = 1.0

    def groups(self) -> list[dict]:
        """Sample groups in display order: campus6, kth, imported."""
        titles = {
            "campus6": "Campus6 初始基线",
            "kth": "KTH 行为数据集",
            "imported": "导入视频",
        }
        return [
            {
                "key": key,
                "title": titles[key],
                "rows": [
                    {
                        "id": row["id"],
                        "label": row.get("label", ""),
                        "person": row.get("person"),
                        "scenario": row.get("scenario"),
                        "split": row.get("split"),
                    }
                    for row in self.selection.get(key) or []
                ],
            }
            for key in ("campus6", "kth", "imported")
        ]

    def get(self, sample_id: str) -> dict:
        try:
            return dict(self._rows[sample_id])
        except KeyError:
            raise KeyError(f"unknown demo sample: {sample_id}") from None

    def _load_eval_scores(self) -> None:
        with self._lock:
            if self._eval_scores is not None:
                return
            path = self.root / "campus6_baseline" / "M1KD.eval_all.pkl"
            with path.open("rb") as stream:
                scores = np.asarray(pickle.load(stream), dtype=np.float32)
            sidecar = path.with_suffix(path.suffix + ".json")
            temperature = 1.0
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                calibration = metadata.get("confidence_calibration") or {}
                if calibration.get("order") == "annotations":
                    temperature = float(calibration.get("temperature", 1.0))
            self._eval_scores = scores
            self._eval_temperature = temperature

    def prediction(self, sample_id: str) -> dict:
        """Return the top-6 calibrated prediction for one demo sample."""
        row = self.get(sample_id)
        prediction_path = row.get("prediction")
        if prediction_path:
            return json.loads(
                (self.root / prediction_path).read_text(encoding="utf-8")
            )
        if row["group"] != "campus6":
            raise KeyError(f"no prediction for {sample_id}")
        self._load_eval_scores()
        index = int(row["annotation_index"])
        probabilities = temperature_scale_probabilities(
            self._eval_scores[index], self._eval_temperature
        )
        ordered = np.argsort(probabilities)[::-1]
        return {
            "schema_version": "protogcn_prediction.v1",
            "status": "completed",
            "sample_id": sample_id,
            "task": "campus6_rtmpose17",
            "label_space_size": len(LABELS),
            "modality": "joint",
            "checkpoint": "models/student/M1KD.int8.pt",
            "checkpoint_format": "portable_int8",
            "model_role": "production student inference",
            "confidence_calibration": {
                "method": "temperature_scaling",
                "temperature": self._eval_temperature,
                "top1_preserved": True,
            },
            "topk": [
                {
                    "class_index": int(class_index),
                    "label": LABELS[int(class_index)],
                    "score": float(probabilities[class_index]),
                }
                for class_index in ordered[:6]
            ],
        }

    def media_path(self, sample_id: str, kind: str) -> Path:
        """Resolve the rgb or pose video file for one sample."""
        row = self.get(sample_id)
        relative = row.get("rgb") if kind == "rgb" else row.get("pose")
        if not relative:
            raise KeyError(f"{kind} media is not ready for {sample_id}")
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def add_imported(self, row: dict) -> None:
        """Persist one finished import into ``selection.json``."""
        with self._lock:
            imported = list(self.selection.setdefault("imported", []))
            imported.append(row)
            self.selection["imported"] = imported
            temporary = self.root / "selection.json.tmp"
            temporary.write_text(
                json.dumps(self.selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.root / "selection.json")
            self._rows[str(row["id"])] = {**row, "group": "imported"}
