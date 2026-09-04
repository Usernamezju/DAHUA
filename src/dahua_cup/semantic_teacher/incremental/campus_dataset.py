"""Auditable Campus6 incremental-dataset lifecycle.

This module deliberately contains no model code.  It is responsible for the
data transaction around one *human-confirmed* hard-sample batch:

``campus_increment (50) -> round snapshot/train input -> campus_all -> clear``.

The move to ``campus_all`` happens only after the caller reports a successful
training process.  Thus a failed run leaves its reviewed examples intact and
reproducible in ``campus_increment``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from dahua_cup.pipeline.build_annotations import LABELS, LABEL_TO_ID


SCHEMA_VERSION = "campus_increment.v1"
DEFAULT_BATCH_SIZE = 50
DEFAULT_REPLAY_SIZE = 250
DEFAULT_SEED = 20260827


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    readable = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)
    return "{}__{}".format(readable.strip("._")[:80] or "sample", digest)


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_json(path: Path, default: Mapping | None = None) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        value = default or {}
    return dict(value) if isinstance(value, Mapping) else dict(default or {})


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_records(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _load_annotation(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("annotations"), list):
        raise ValueError("invalid Campus6 annotation file: {}".format(path))
    split = value.get("split") or {}
    if not all(name in split for name in ("train", "val", "test")):
        raise ValueError("Campus6 annotation is missing train/val/test splits")
    return value


def _write_annotation(path: Path, annotations: list[dict], split: Mapping[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": {name: list(split.get(name, [])) for name in ("train", "val", "test")},
        "annotations": annotations,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _split_records(rows: list[dict], *, seed: int) -> dict[str, str]:
    """Stable group-stratified 70/15/15 assignment with train priority.

    A sparse class never loses all of its new examples from training.  Samples
    from the same ``source_group`` stay together, exactly as for the original
    Campus6 builder.
    """
    by_label: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_label[str(row["label"])][str(row["source_group"])].append(row)
    assignment: dict[str, str] = {}
    for label in LABELS:
        groups = list(by_label.get(label, {}))
        groups.sort(key=lambda group: hashlib.sha256(
            (str(seed) + "|" + label + "|" + group).encode("utf-8")
        ).hexdigest())
        count = len(groups)
        if count <= 1:
            val_count = test_count = 0
        elif count == 2:
            # Keep one training group; the second is held out deterministically.
            val_count, test_count = 0, 1
        else:
            test_count = max(1, round(count * 0.15))
            val_count = max(1, round(count * 0.15))
            while test_count + val_count > count - 1:
                if test_count >= val_count:
                    test_count -= 1
                else:
                    val_count -= 1
        for index, group in enumerate(groups):
            split = "test" if index < test_count else "val" if index < test_count + val_count else "train"
            for row in by_label[label][group]:
                assignment[str(row["sample_id"])] = split
    return assignment


def _select_replay_annotations(
    annotation: Mapping,
    *,
    limit: int = DEFAULT_REPLAY_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Pick a deterministic, class-balanced replay set from historical train.

    Only prior *training* examples are eligible.  This keeps the baseline
    validation/test partitions out of both optimisation and replay selection,
    while making every incremental training input a bounded ``50 + 250``
    dataset (or smaller when the historical training set has fewer examples).
    """
    if limit < 1:
        return []
    train_ids = {str(value) for value in annotation.get("split", {}).get("train", [])}
    by_label: dict[int, list[dict]] = defaultdict(list)
    for item in annotation.get("annotations", []):
        if str(item.get("frame_dir")) not in train_ids:
            continue
        try:
            label = int(item["label"])
        except (KeyError, TypeError, ValueError):
            continue
        by_label[label].append(item)

    # A stable hash ordering makes the replay input reproducible across
    # processes and avoids favouring the order in which old records were
    # appended to campus_all.
    for label, rows in by_label.items():
        rows.sort(key=lambda item: hashlib.sha256(
            (str(seed) + "|" + str(label) + "|" + str(item.get("frame_dir"))).encode("utf-8")
        ).hexdigest())
    selected: list[dict] = []
    offsets = {label: 0 for label in sorted(by_label)}
    while len(selected) < limit:
        progressed = False
        for label in sorted(by_label):
            offset = offsets[label]
            rows = by_label[label]
            if offset >= len(rows):
                continue
            selected.append(rows[offset])
            offsets[label] = offset + 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _annotation_from_record(row: Mapping, feature_path: Path) -> dict:
    with np.load(feature_path, allow_pickle=False) as artifact:
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)
        score = np.asarray(artifact["keypoint_score"], dtype=np.float32)
        total_frames = int(np.asarray(artifact.get("total_frames", keypoint.shape[1])).item())
    if keypoint.ndim != 4 or keypoint.shape[2:] != (17, 2):
        raise ValueError("{} must contain [M,T,17,2] COCO-17 keypoints; got {}".format(feature_path, keypoint.shape))
    if score.shape != keypoint.shape[:3]:
        raise ValueError("{} has incompatible keypoint_score {}".format(feature_path, score.shape))
    if not np.isfinite(keypoint).all() or not np.isfinite(score).all():
        raise ValueError("{} contains non-finite pose values".format(feature_path))
    return {
        "frame_dir": str(row["sample_id"]),
        "total_frames": total_frames,
        "label": LABEL_TO_ID[str(row["label"])],
        "keypoint": keypoint,
        "keypoint_score": score,
        "img_shape": (1, 1),
    }


@dataclass(frozen=True)
class IncrementalRound:
    round_id: str
    root: Path
    incoming_annotation: Path
    training_annotation: Path
    sample_ids: tuple[str, ...]


class CampusIncrementalDataset:
    """Filesystem transaction manager for Campus6 incremental data."""

    def __init__(
        self,
        dataset_root: Path,
        runtime_root: Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset_root = Path(dataset_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.batch_size = batch_size
        self.seed = seed
        self.baseline_root = self.dataset_root / "campus6_baseline"
        self.increment_root = self.dataset_root / "campus_increment"
        self.all_root = self.dataset_root / "campus_all"
        self.rounds_root = self.runtime_root / "incremental" / "rounds"

    @property
    def increment_records_path(self) -> Path:
        return self.increment_root / "records.jsonl"

    @property
    def increment_state_path(self) -> Path:
        return self.increment_root / "state.json"

    def ensure_layout(self) -> None:
        for root in (self.increment_root, self.all_root):
            (root / "features").mkdir(parents=True, exist_ok=True)
            (root / "videos").mkdir(parents=True, exist_ok=True)
        self.rounds_root.mkdir(parents=True, exist_ok=True)
        baseline = self.baseline_root / "annotations_with_all.pkl"
        all_annotation = self.all_root / "annotations_with_all.pkl"
        if not all_annotation.exists() and baseline.is_file():
            shutil.copy2(baseline, all_annotation)
            baseline_summary = baseline.with_suffix(".summary.json")
            if baseline_summary.is_file():
                shutil.copy2(baseline_summary, self.all_root / baseline_summary.name)
            _atomic_json(self.all_root / "state.json", {
                "schema_version": SCHEMA_VERSION,
                "initialized_from": str(baseline),
                "initialized_at": _now(),
                "accepted_rounds": [],
            })
        if not self.increment_state_path.exists():
            _atomic_json(self.increment_state_path, {
                "schema_version": SCHEMA_VERSION,
                "state": "collecting",
                "batch_size": self.batch_size,
                "seed": self.seed,
                "staged_sample_ids": [],
            })
        increment_annotation = self.increment_root / "annotations_with_all.pkl"
        if not increment_annotation.exists():
            _write_annotation(
                increment_annotation, [], {"train": [], "val": [], "test": []}
            )

    def records(self) -> list[dict]:
        return _load_records(self.increment_records_path)

    def status(self) -> dict:
        self.ensure_layout()
        records = self.records()
        state = _load_json(self.increment_state_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "state": state.get("state", "collecting"),
            "batch_size": self.batch_size,
            "staged_sample_count": len(records),
            "remaining_to_train": max(0, self.batch_size - len(records)),
            "ready": len(records) >= self.batch_size,
            "labels": dict(sorted(Counter(str(row["label"]) for row in records).items())),
            "active_round_id": state.get("active_round_id"),
        }

    def stage(
        self,
        sample: Mapping,
        *,
        feature_path: Path,
        video_path: Optional[Path] = None,
    ) -> dict:
        """Copy one approved pose artifact into ``campus_increment``.

        This operation is idempotent by sample id.  It does not alter the
        original review record nor re-split already staged samples.  ``video_path``
        is retained only for call compatibility and intentionally ignored:
        pose-only incremental data must not retain raw RGB video.
        """
        self.ensure_layout()
        sample_id = str(sample.get("sample_id") or "").strip()
        label = str(sample.get("manual_label") or "").strip()
        if not sample_id or label not in LABEL_TO_ID:
            raise ValueError("only a human-reviewed Campus6 label can be staged")
        feature_path = Path(feature_path)
        if not feature_path.is_file():
            raise FileNotFoundError("approved sample has no COCO-17 pose artifact")
        rows = self.records()
        existing = next((row for row in rows if row.get("sample_id") == sample_id), None)
        if existing:
            return {"staged": False, "reason": "already_staged", **self.status()}
        name = _safe_name(sample_id)
        target_feature = self.increment_root / "features" / (name + ".npz")
        shutil.copy2(feature_path, target_feature)
        # Validate before recording the item, so a corrupt artifact can never
        # advance the automatic 50-sample threshold.
        record = {
            "sample_id": sample_id,
            "label": label,
            "source_group": "incremental/" + sample_id,
            "feature": str(target_feature.relative_to(self.increment_root)),
            "video": "",
            "reviewer": str(sample.get("reviewer") or ""),
            "reviewed_at": str(sample.get("reviewed_at") or _now()),
            "incremental_reason": str(sample.get("incremental_reason") or ""),
            "staged_at": _now(),
        }
        _annotation_from_record(record, target_feature)
        rows.append(record)
        _write_records(self.increment_records_path, rows)
        self._write_increment_snapshot(rows)
        state = _load_json(self.increment_state_path)
        state.update({"state": "ready" if len(rows) >= self.batch_size else "collecting", "staged_sample_ids": [row["sample_id"] for row in rows], "updated_at": _now()})
        _atomic_json(self.increment_state_path, state)
        return {"staged": True, **self.status()}

    def _write_increment_snapshot(self, rows: list[dict]) -> None:
        """Maintain a baseline-shaped, provisional dataset for inspection.

        The split is recomputed only for the accumulating workspace.  The
        immutable split used for a training round is separately frozen in its
        round directory by :meth:`prepare_round`.
        """
        annotations = []
        split = {"train": [], "val": [], "test": []}
        assignment = _split_records(rows, seed=self.seed)
        for row in rows:
            annotation = _annotation_from_record(
                row, self.increment_root / str(row["feature"])
            )
            annotations.append(annotation)
            split[assignment[row["sample_id"]]].append(annotation["frame_dir"])
        _write_annotation(
            self.increment_root / "annotations_with_all.pkl", annotations, split
        )
        _atomic_json(self.increment_root / "annotations_with_all.summary.json", {
            "schema_version": SCHEMA_VERSION,
            "sample_count": len(annotations),
            "split": {name: len(values) for name, values in split.items()},
            "split_status": "provisional_until_50_sample_round_is_frozen",
            "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
            "updated_at": _now(),
        })

    def prepare_round(self) -> IncrementalRound:
        """Freeze a ready batch and create a fixed-split training input.

        Each round contains exactly the first 50 approved new examples plus
        at most 250 class-balanced examples replayed from historical training
        data.  Later arrivals remain staged for the next round.
        """
        self.ensure_layout()
        state = _load_json(self.increment_state_path)
        if state.get("active_round_id"):
            raise RuntimeError("an incremental round is already active")
        rows = self.records()
        if len(rows) < self.batch_size:
            raise RuntimeError("incremental batch is not ready")
        batch = rows[: self.batch_size]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        round_id = "campus6_increment_{}_{}".format(timestamp, hashlib.sha256(
            "|".join(row["sample_id"] for row in batch).encode("utf-8")
        ).hexdigest()[:8])
        root = self.rounds_root / round_id
        incoming_root = root / "incoming"
        incoming_root.mkdir(parents=True, exist_ok=False)
        assignment = _split_records(batch, seed=self.seed)
        annotations = []
        split = {"train": [], "val": [], "test": []}
        for row in batch:
            feature = self.increment_root / str(row["feature"])
            annotation = _annotation_from_record(row, feature)
            annotations.append(annotation)
            split[assignment[row["sample_id"]]].append(annotation["frame_dir"])
        incoming_annotation = incoming_root / "annotations_with_all.pkl"
        _write_annotation(incoming_annotation, annotations, split)
        _write_records(incoming_root / "records.jsonl", batch)
        report = {
            "schema_version": SCHEMA_VERSION,
            "round_id": round_id,
            "sample_count": len(batch),
            "split": {name: len(values) for name, values in split.items()},
            "split_by_label": {
                label: {name: sum(1 for item in annotations if item["label"] == LABEL_TO_ID[label] and item["frame_dir"] in split[name]) for name in split}
                for label in LABELS
            },
            "created_at": _now(),
        }
        _atomic_json(root / "round.json", report)
        training_annotation = root / "training_annotations_with_all.pkl"
        all_annotation = self.all_root / "annotations_with_all.pkl"
        if not all_annotation.is_file():
            raise FileNotFoundError("campus_all is not initialized from campus6_baseline")
        previous = _load_annotation(all_annotation)
        previous_ids = {str(item["frame_dir"]) for item in previous["annotations"]}
        duplicate = previous_ids.intersection(row["sample_id"] for row in batch)
        if duplicate:
            raise ValueError("sample already exists in campus_all: {}".format(sorted(duplicate)[0]))
        replay_annotations = _select_replay_annotations(
            previous, limit=DEFAULT_REPLAY_SIZE, seed=self.seed
        )
        replay_ids = [str(item["frame_dir"]) for item in replay_annotations]
        training_split = {
            "train": replay_ids + list(split["train"]),
            "val": list(split["val"]),
            "test": list(split["test"]),
        }
        _write_annotation(
            training_annotation,
            replay_annotations + annotations,
            training_split,
        )
        report.update({
            "replay_sample_count": len(replay_annotations),
            "training_sample_count": len(replay_annotations) + len(annotations),
            "replay_labels": dict(sorted(Counter(
                str(item["label"]) for item in replay_annotations
            ).items())),
        })
        _atomic_json(root / "round.json", report)
        state.update({"state": "training", "active_round_id": round_id, "active_sample_ids": [row["sample_id"] for row in batch], "updated_at": _now()})
        _atomic_json(self.increment_state_path, state)
        return IncrementalRound(round_id, root, incoming_annotation, training_annotation, tuple(row["sample_id"] for row in batch))

    def complete_round(self, round_: IncrementalRound, *, training_metadata: Optional[Mapping] = None) -> dict:
        """Commit an already-successful round, then clear only its 50 rows."""
        self.ensure_layout()
        state = _load_json(self.increment_state_path)
        if state.get("active_round_id") != round_.round_id:
            raise ValueError("round is not the currently active incremental batch")
        incoming = _load_annotation(round_.incoming_annotation)
        current_path = self.all_root / "annotations_with_all.pkl"
        current = _load_annotation(current_path)
        existing = {str(item["frame_dir"]) for item in current["annotations"]}
        if existing.intersection(round_.sample_ids):
            raise ValueError("cannot merge: an incoming sample already exists in campus_all")
        merged_annotations = list(current["annotations"]) + list(incoming["annotations"])
        merged_split = {name: list(current["split"].get(name, [])) + list(incoming["split"].get(name, [])) for name in ("train", "val", "test")}
        _write_annotation(current_path, merged_annotations, merged_split)
        all_state_path = self.all_root / "state.json"
        all_state = _load_json(all_state_path, {"schema_version": SCHEMA_VERSION, "accepted_rounds": []})
        accepted = list(all_state.get("accepted_rounds") or [])
        accepted.append({"round_id": round_.round_id, "sample_count": len(round_.sample_ids), "completed_at": _now(), "training": dict(training_metadata or {})})
        all_state.update({"schema_version": SCHEMA_VERSION, "accepted_rounds": accepted, "updated_at": _now()})
        _atomic_json(all_state_path, all_state)
        staged_rows = self.records()
        completed_ids = set(round_.sample_ids)
        remaining = [row for row in staged_rows if row.get("sample_id") not in completed_ids]
        for relative in ("records.jsonl", "state.json", "annotations_with_all.pkl", "annotations_with_all.summary.json"):
            path = self.increment_root / relative
            if path.exists() and path.name != "state.json":
                path.unlink()
        # Reviews can arrive while a batch is training.  Remove only the
        # completed batch's files; preserving later arrivals prevents an
        # otherwise invisible data loss when the count briefly exceeds 50.
        completed_rows = [row for row in staged_rows if row.get("sample_id") in completed_ids]
        for row in completed_rows:
            for field in ("feature", "video"):
                relative = str(row.get(field) or "")
                if relative:
                    (self.increment_root / relative).unlink(missing_ok=True)
        for directory in (self.increment_root / "features", self.increment_root / "videos"):
            directory.mkdir(parents=True, exist_ok=True)
        _write_records(self.increment_records_path, remaining)
        self._write_increment_snapshot(remaining)
        next_state = {"schema_version": SCHEMA_VERSION, "state": "ready" if len(remaining) >= self.batch_size else "collecting", "batch_size": self.batch_size, "seed": self.seed, "staged_sample_ids": [row["sample_id"] for row in remaining], "last_completed_round": round_.round_id, "updated_at": _now()}
        _atomic_json(self.increment_state_path, next_state)
        result = self.status()
        result.update({"round_id": round_.round_id, "merged_sample_count": len(round_.sample_ids), "campus_all_sample_count": len(merged_annotations)})
        return result

    def fail_round(self, round_: IncrementalRound, message: str) -> None:
        state = _load_json(self.increment_state_path)
        if state.get("active_round_id") == round_.round_id:
            state.update({"state": "failed", "failure": str(message)[:1000], "updated_at": _now()})
            _atomic_json(self.increment_state_path, state)
        _atomic_json(round_.root / "failure.json", {"round_id": round_.round_id, "failed_at": _now(), "message": str(message)[:4000]})
