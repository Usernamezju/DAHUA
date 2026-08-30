"""Automatic NTU120 pseudo-label, mining, distillation and promotion loop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from dahua_cup.pipeline.collect_ntu120_pseudo import (
    NTU120PseudoThresholds,
    collect_records,
    load_labels,
    merge_records,
)
from dahua_cup.pipeline.common import file_hash, read_jsonl, write_jsonl
from dahua_cup.pipeline.mine_candidates import (
    class_statistics,
    load_mining_configuration,
    mine,
    read_prediction_directory,
    select_budget,
)
from dahua_cup.semantic_teacher.incremental.model_registry import ModelRegistry
from dahua_cup.semantic_teacher.incremental.release_gate import ProductionPointer

if TYPE_CHECKING:
    from .config import Settings
    from .jobs import JobManager
    from .store import ReviewStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_ntu120_release(
    candidate: dict,
    baseline: dict,
    *,
    candidate_size_bytes: int,
    maximum_checkpoint_bytes: int,
    minimum_mean_class_improvement: float,
    maximum_top1_drop: float,
) -> dict:
    """Apply the non-regression gates used before an automatic promotion."""
    required = {"top1_acc", "top5_acc", "mean_class_accuracy"}
    missing = sorted(required - candidate.keys())
    missing += sorted(
        f"baseline.{name}" for name in required - baseline.keys()
    )
    if missing:
        raise ValueError("missing NTU120 release metrics: " + ", ".join(missing))
    checks = {
        "mean_class_improved": (
            candidate["mean_class_accuracy"]
            >= baseline["mean_class_accuracy"]
            + minimum_mean_class_improvement
        ),
        "top1_not_regressed": (
            candidate["top1_acc"]
            >= baseline["top1_acc"] - maximum_top1_drop
        ),
        "checkpoint_size": (
            int(candidate_size_bytes) <= int(maximum_checkpoint_bytes)
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


@dataclass(frozen=True)
class EvolutionConfiguration:
    enabled: bool
    scan_interval_seconds: int
    pseudo_thresholds: NTU120PseudoThresholds
    review_priority: int
    hard_mining_enabled: bool
    hard_mining_config: Path
    training_enabled: bool
    minimum_accepted: int
    minimum_new: int
    cooldown_hours: float
    epochs: int
    learning_rate: float
    gpus: int
    max_replay_samples: int
    max_pseudo_samples: int
    maximum_checkpoint_bytes: int
    minimum_mean_class_improvement: float
    maximum_top1_drop: float
    base_annotation_candidates: tuple[Path, ...]
    python_candidates: tuple[Path, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        data_root: Path,
        repository_root: Path,
    ) -> "EvolutionConfiguration":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"auto-evolution config not found: {source}")
        value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if value.get("schema_version") != "auto_evolution_ntu120.v1":
            raise ValueError("unsupported auto-evolution config schema")
        pseudo = dict(value.get("pseudo_label") or {})
        hard = dict(value.get("hard_mining") or {})
        training = dict(value.get("training") or {})

        def expand(raw: str) -> Path:
            rendered = str(raw).format(
                data_root=data_root,
                repository_root=repository_root,
                python=sys.executable,
            )
            return Path(rendered).expanduser().resolve()

        hard_path = Path(str(hard.get("config", "hard_mining.yaml")))
        if not hard_path.is_absolute():
            hard_path = source.parent / hard_path
        result = cls(
            enabled=bool(value.get("enabled", True)),
            scan_interval_seconds=max(
                5, int(value.get("scan_interval_seconds", 30))
            ),
            pseudo_thresholds=NTU120PseudoThresholds(
                accept_score=float(pseudo.get("accept_score", 0.78)),
                review_score=float(pseudo.get("review_score", 0.50)),
                minimum_teacher_confidence=float(
                    pseudo.get("minimum_teacher_confidence", 0.60)
                ),
                minimum_pose_quality=float(
                    pseudo.get("minimum_pose_quality", 0.70)
                ),
            ),
            review_priority=max(0, int(pseudo.get("review_priority", 750000))),
            hard_mining_enabled=bool(hard.get("enabled", True)),
            hard_mining_config=hard_path.resolve(),
            training_enabled=bool(training.get("enabled", True)),
            minimum_accepted=max(
                1,
                int(training.get("minimum_accepted_pseudo_labels", 20)),
            ),
            minimum_new=max(
                1, int(training.get("minimum_new_pseudo_labels", 10))
            ),
            cooldown_hours=max(0.0, float(training.get("cooldown_hours", 12))),
            epochs=max(1, int(training.get("epochs", 10))),
            learning_rate=float(training.get("learning_rate", 0.0005)),
            gpus=max(1, int(training.get("gpus", 1))),
            max_replay_samples=max(
                0, int(training.get("max_replay_samples", 4000))
            ),
            max_pseudo_samples=max(
                1, int(training.get("max_pseudo_samples", 2000))
            ),
            maximum_checkpoint_bytes=max(
                1, int(training.get("maximum_checkpoint_bytes", 50 * 1024 * 1024))
            ),
            minimum_mean_class_improvement=float(
                training.get(
                    "minimum_mean_class_accuracy_improvement", 0.001
                )
            ),
            maximum_top1_drop=max(
                0.0, float(training.get("maximum_top1_accuracy_drop", 0.002))
            ),
            base_annotation_candidates=tuple(
                expand(item)
                for item in training.get("base_annotation_candidates", ())
            ),
            python_candidates=tuple(
                expand(item) for item in training.get("python_candidates", ())
            ),
        )
        result.pseudo_thresholds.validate()
        if result.learning_rate <= 0:
            raise ValueError("auto-evolution learning_rate must be positive")
        return result


class EvolutionManager:
    """A single background controller; model training is never concurrent."""

    def __init__(
        self,
        settings: "Settings",
        store: "ReviewStore",
        jobs: "JobManager",
    ):
        self.settings = settings
        self.store = store
        self.jobs = jobs
        if not all(
            (
                settings.evolution_config,
                settings.evolution_root,
                settings.model_registry_path,
                settings.production_pointer_path,
                settings.ntu120_label_map,
            )
        ):
            raise ValueError("NTU120 auto-evolution paths are incomplete")
        self.root = settings.evolution_root
        self.config = EvolutionConfiguration.load(
            settings.evolution_config,
            data_root=settings.data_root,
            repository_root=settings.repository_root,
        )
        self.state_path = self.root / "state.json"
        self.pseudo_path = self.root / "pseudo" / "ntu120_pseudo_labels.jsonl"
        self.pseudo_history_path = (
            self.root / "pseudo" / "ntu120_pseudo_history.jsonl"
        )
        self.hard_path = self.root / "hard_mining" / "selected.jsonl"
        self.hard_stats_path = self.root / "hard_mining" / "class_stats.json"
        self.registry_path = settings.model_registry_path
        self.pointer_path = settings.production_pointer_path
        self.stop_event = threading.Event()
        self.run_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self._state = self._load_state()
        self._state["enabled"] = self.config.enabled

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                pass
        return {
            "schema_version": "auto_evolution_state.v1",
            "enabled": self.config.enabled,
            "status": "starting" if self.config.enabled else "disabled",
            "stage": "idle",
            "last_scan_at": None,
            "last_error": "",
            "pseudo": {"accepted": 0, "review": 0, "rejected": 0},
            "hard_mining": {"selected": 0},
            "training": {
                "eligible": False,
                "reasons": [],
                "attempted_fingerprints": [],
                "last_attempt_at": None,
                "last_report": None,
            },
        }

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self._state)

    def status(self) -> dict:
        # Status must stay responsive while a multi-hour training run holds
        # ``run_lock``. The controller only replaces JSON-compatible values.
        value = json.loads(json.dumps(self._state))
        value["configuration"] = {
            **asdict(self.config),
            "pseudo_thresholds": asdict(self.config.pseudo_thresholds),
            "hard_mining_config": str(self.config.hard_mining_config),
            "base_annotation_candidates": [
                str(path) for path in self.config.base_annotation_candidates
            ],
            "python_candidates": [
                str(path) for path in self.config.python_candidates
            ],
        }
        value["paths"] = {
            "root": str(self.root),
            "pseudo_labels": str(self.pseudo_path),
            "pseudo_history": str(self.pseudo_history_path),
            "hard_samples": str(self.hard_path),
            "registry": str(self.registry_path),
            "production_pointer": str(self.pointer_path),
        }
        value["active_checkpoint"] = str(
            self.settings.resolve_student_checkpoint() or ""
        )
        return value

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.config.enabled:
            self._state["status"] = "disabled"
            self._save_state()
            return
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._loop,
            name="dahua-auto-evolution",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def trigger(self) -> bool:
        if not self.config.enabled or not self.run_lock.acquire(blocking=False):
            return False
        threading.Thread(
            target=self._run_once_with_acquired_lock,
            name="dahua-auto-evolution-manual",
            daemon=True,
        ).start()
        return True

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # keep the service alive and expose state
                self._state["status"] = "error"
                self._state["stage"] = "idle"
                self._state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._save_state()
            self.stop_event.wait(self.config.scan_interval_seconds)

    def run_once(self) -> None:
        if not self.run_lock.acquire(blocking=False):
            return
        self._run_once_with_acquired_lock()

    def _run_once_with_acquired_lock(self) -> None:
        try:
            self._state["status"] = "running"
            self._state["last_error"] = ""
            self._collect_pseudo_labels()
            if self.config.hard_mining_enabled:
                self._mine_hard_samples()
            self._evaluate_training_trigger()
            self._state["last_scan_at"] = _now()
            self._state["stage"] = "idle"
            self._state["status"] = "watching"
            self._save_state()
        finally:
            self.run_lock.release()

    def _collect_pseudo_labels(self) -> None:
        self._state["stage"] = "pseudo_label_collection"
        labels = load_labels(self.settings.ntu120_label_map)
        prediction_dir = self.settings.artifact_root / "predictions"
        teacher_dir = self.settings.artifact_root / "teachers"
        existing = read_jsonl(self.pseudo_path) if self.pseudo_path.is_file() else []
        current = collect_records(
            prediction_dir,
            teacher_dir,
            labels,
            self.config.pseudo_thresholds,
        )
        previous_by_id = {row["sample_id"]: row for row in existing}
        merged = merge_records(existing, current)
        if merged != existing:
            write_jsonl(self.pseudo_path, merged)
        merged_by_id = {row["sample_id"]: row for row in merged}
        changed = [
            merged_by_id[row["sample_id"]]
            for row in current
            if previous_by_id.get(row["sample_id"], {}).get("fingerprint")
            != row.get("fingerprint")
        ]
        _append_jsonl(self.pseudo_history_path, changed)
        for row in changed:
            if row["status"] != "review":
                continue
            try:
                self.store.escalate_hard_sample(
                    row["sample_id"],
                    hard_score=float(row["quality_score"]),
                    priority=self.config.review_priority,
                    reason=(
                        row["review_reasons"][0]
                        if row["review_reasons"]
                        else "ntu120_pseudo_review"
                    ),
                    payload={"ntu120_pseudo_record": row},
                )
            except KeyError:
                continue
        self._state["pseudo"] = {
            status: sum(row["status"] == status for row in merged)
            for status in ("accepted", "review", "rejected")
        }
        self._state["pseudo"]["changed"] = len(changed)

    def _mine_hard_samples(self) -> None:
        self._state["stage"] = "hard_sample_mining"
        prediction_dir = self.settings.artifact_root / "predictions"
        if not prediction_dir.is_dir() or not any(prediction_dir.glob("*.json")):
            self._state["hard_mining"] = {"selected": 0}
            return
        artifact_paths = sorted(prediction_dir.glob("*.json"))
        teacher_dir = self.settings.artifact_root / "teachers"
        if teacher_dir.is_dir():
            artifact_paths.extend(sorted(teacher_dir.glob("*.json")))
        input_fingerprint = hashlib.sha256(
            "\n".join(
                f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
                for path in artifact_paths
            ).encode("utf-8")
        ).hexdigest()
        if (
            self._state.get("hard_mining", {}).get("input_fingerprint")
            == input_fingerprint
        ):
            return
        configured = load_mining_configuration(self.config.hard_mining_config)
        students = [
            row for row in read_prediction_directory(prediction_dir)
            if row.get("topk")
            and row.get("status", "completed") == "completed"
        ]
        if not students:
            self._state["hard_mining"] = {
                "selected": 0,
                "input_fingerprint": input_fingerprint,
            }
            return
        teachers = {}
        if teacher_dir.is_dir() and any(teacher_dir.glob("*.json")):
            for row in read_prediction_directory(teacher_dir):
                value = row.get("result") or row
                sample_id = value.get("sample_id") or row.get("sample_id")
                if sample_id:
                    teachers[str(sample_id)] = row
        ranked = mine(
            students,
            teachers,
            rare_class_quantile=configured["rare_class_quantile"],
            label_space_size=120,
            before_teacher_weights=configured["before_teacher_weights"],
            after_teacher_weights=configured["after_teacher_weights"],
            reason_threshold=configured["reason_threshold"],
        )
        selected = select_budget(
            ranked,
            limit=configured["limit"],
            per_class_limit=configured["per_class_limit"],
            minimum_score=configured["minimum_score"],
        )
        write_jsonl(self.hard_path, selected)
        statistics = class_statistics(
            [row["predicted_label"] for row in ranked],
            configured["rare_class_quantile"],
        )
        _atomic_json(self.hard_stats_path, statistics)
        for row in selected:
            try:
                self.store.escalate_hard_sample(
                    str(row["sample_id"]),
                    hard_score=float(row["hard_score"]),
                    priority=int(row["review_priority"]),
                    reason=(
                        row["hard_reasons"][0]
                        if row["hard_reasons"] else "offline_hard_mining"
                    ),
                    payload={"hard_components": row["hard_components"]},
                )
            except KeyError:
                continue
        self._state["hard_mining"] = {
            "selected": len(selected),
            "observed_classes": statistics.get("observed_class_count", 0),
            "rare_labels": statistics.get("rare_labels", []),
            "input_fingerprint": input_fingerprint,
        }

    def _resolve_training_python(self) -> Path | None:
        return next(
            (
                path for path in self.config.python_candidates
                if path.is_file() and os.access(path, os.X_OK)
            ),
            None,
        )

    def _resolve_base_annotation(self) -> Path | None:
        return next(
            (
                path for path in self.config.base_annotation_candidates
                if path.is_file()
            ),
            None,
        )

    def _evaluate_training_trigger(self) -> None:
        self._state["stage"] = "training_trigger"
        accepted = [
            row for row in (
                read_jsonl(self.pseudo_path) if self.pseudo_path.is_file() else []
            )
            if row.get("status") == "accepted"
        ]
        trainable = [
            row
            for row in accepted
            if Path(str(row.get("feature_path", ""))).expanduser().is_file()
        ]
        training_state = self._state.setdefault("training", {})
        attempted = set(training_state.get("attempted_fingerprints") or [])
        new_rows = [
            row for row in trainable if row.get("fingerprint") not in attempted
        ]
        reasons = []
        if not self.config.training_enabled:
            reasons.append("training_disabled")
        if len(trainable) < self.config.minimum_accepted:
            reasons.append("insufficient_accepted_pseudo_labels")
        if len(new_rows) < self.config.minimum_new:
            reasons.append("insufficient_new_pseudo_labels")
        base_annotation = self._resolve_base_annotation()
        if base_annotation is None:
            reasons.append("missing_ntu120_3d_base_annotation")
        training_python = self._resolve_training_python()
        if training_python is None:
            reasons.append("missing_training_python")
        gpu_state = self.jobs.gpus.state()
        available_gpu_ids = {
            int(row["index"]) for row in gpu_state.get("gpus", [])
        }
        training_gpu_ids = [
            int(gpu_id)
            for gpu_id in gpu_state.get("student_gpu_ids", [])
            if int(gpu_id) in available_gpu_ids
        ][: self.config.gpus]
        if len(training_gpu_ids) < self.config.gpus:
            reasons.append("insufficient_selected_training_gpus")
        base_checkpoint = self.settings.resolve_student_checkpoint()
        if base_checkpoint is None or not base_checkpoint.is_file():
            reasons.append("missing_production_checkpoint")
        last_attempt = training_state.get("last_attempt_epoch")
        if last_attempt is not None:
            elapsed_hours = (time.time() - float(last_attempt)) / 3600.0
            if elapsed_hours < self.config.cooldown_hours:
                reasons.append("training_cooldown")
        training_state.update(
            {
                "eligible": not reasons,
                "reasons": reasons,
                "accepted_count": len(accepted),
                "trainable_count": len(trainable),
                "missing_feature_count": len(accepted) - len(trainable),
                "new_count": len(new_rows),
                "base_annotation": str(base_annotation or ""),
                "training_python": str(training_python or ""),
                "training_gpu_ids": training_gpu_ids,
                "base_checkpoint": str(base_checkpoint or ""),
            }
        )
        if reasons:
            return
        self._train_candidate(
            trainable,
            base_annotation,
            training_python,
            base_checkpoint,
        )

    def _run_process(
        self,
        command: list[str],
        *,
        environment: dict | None = None,
    ) -> tuple[str, float]:
        env = os.environ.copy()
        if environment:
            env.update(environment)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=self.settings.repository_root,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout, time.monotonic() - started

    @staticmethod
    def _metrics(output: str) -> dict:
        patterns = {
            "top1_acc": r"\btop1_acc:\s*([0-9.]+)",
            "top5_acc": r"\btop5_acc:\s*([0-9.]+)",
            "mean_class_accuracy": r"\bmean_class_accuracy:\s*([0-9.]+)",
        }
        metrics = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                metrics[name] = float(matches[-1])
        if set(metrics) != set(patterns):
            raise ValueError(
                "ProtoGCN evaluation output is missing metrics: "
                + ", ".join(sorted(set(patterns) - set(metrics)))
            )
        return metrics

    def _evaluate_checkpoint(
        self,
        training_python: Path,
        annotation: Path,
        checkpoint: Path,
        work_dir: Path,
        name: str,
    ) -> dict:
        config = (
            self.settings.repository_root
            / "dahua_cup"
            / "configs"
            / "protogcn"
            / "ntu120_ntu25_bone_distill.py"
        )
        output_path = work_dir / f"{name}_predictions.pkl"
        environment = {
            "DAHUA_NTU120_DISTILL_ANN": str(annotation),
            "DAHUA_NTU120_DISTILL_WORK_DIR": str(work_dir),
            "PATH": str(training_python.parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.settings.repository_root),
        }
        output, seconds = self._run_process(
            [
                "bash",
                "gcn_models/ProtoGCN/tools/dist_test.sh",
                str(config),
                str(checkpoint),
                "1",
                "--out",
                str(output_path),
            ],
            environment=environment,
        )
        metrics = self._metrics(output)
        metrics["evaluation_seconds"] = seconds
        return metrics

    def _ensure_baseline_pointer(
        self,
        registry: ModelRegistry,
        pointer: ProductionPointer,
        checkpoint: Path,
        metrics: dict,
    ) -> str:
        """Register the pre-existing model so the first promotion is reversible."""
        current = pointer.read().get("current_model_id")
        if current:
            return str(current)
        checkpoint_hash = _sha256(checkpoint)
        model_id = f"protogcn-ntu120-baseline-{checkpoint_hash[:12]}"
        history = [
            row for row in registry.records()
            if row.get("model_id") == model_id
        ]
        if not history:
            registry.register(
                {
                    "model_id": model_id,
                    "dataset_id": "ntu120-official-baseline",
                    "checkpoint_path": str(checkpoint),
                    "config_path": str(
                        self.settings.repository_root
                        / "gcn_models"
                        / "ProtoGCN"
                        / "configs"
                        / "ntu120_xsub"
                        / "b.py"
                    ),
                    "config_hash": file_hash(
                        self.settings.repository_root
                        / "gcn_models"
                        / "ProtoGCN"
                        / "configs"
                        / "ntu120_xsub"
                        / "b.py"
                    ),
                    "checkpoint_hash": checkpoint_hash,
                    "metrics": metrics,
                    "edge_size_bytes": checkpoint.stat().st_size,
                    "latency": {
                        "device": "not_measured",
                        "milliseconds": None,
                    },
                    "source": "pre_existing_production_checkpoint",
                }
            )
            registry.transition(model_id, "validated")
            registry.transition(model_id, "canary")
            registry.transition(model_id, "production")
        elif history[-1].get("status") != "production":
            raise ValueError(
                "existing baseline registry record is not production: "
                f"{model_id}"
            )
        pointer.promote(
            model_id,
            {
                "passed": True,
                "reason": "register_pre_existing_production_checkpoint",
            },
        )
        return model_id

    def _train_candidate(
        self,
        accepted: list[dict],
        base_annotation: Path,
        training_python: Path,
        base_checkpoint: Path,
    ) -> None:
        training_state = self._state["training"]
        training_state["last_attempt_at"] = _now()
        training_state["last_attempt_epoch"] = time.time()
        training_state["last_attempt_fingerprints"] = sorted(
            {str(row.get("fingerprint", "")) for row in accepted}
        )
        self._save_state()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / "training" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        annotation_path = run_dir / "ntu120_distill.pkl"
        run_pseudo_path = run_dir / "accepted_pseudo_labels.jsonl"
        write_jsonl(run_pseudo_path, accepted)
        environment = {
            "PATH": str(training_python.parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.settings.repository_root),
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(value)
                for value in training_state["training_gpu_ids"]
            ),
        }
        self._state["stage"] = "distillation_dataset_build"
        build_output, _ = self._run_process(
            [
                str(training_python),
                "-m",
                "dahua_cup.pipeline.build_ntu120_distill_dataset",
                "--base-annotation",
                str(base_annotation),
                "--pseudo-jsonl",
                str(run_pseudo_path),
                "--label-map",
                str(self.settings.ntu120_label_map),
                "--output",
                str(annotation_path),
                "--max-replay-samples",
                str(self.config.max_replay_samples),
                "--max-pseudo-samples",
                str(self.config.max_pseudo_samples),
            ],
            environment=environment,
        )
        (run_dir / "dataset_build.log").write_text(
            build_output, encoding="utf-8"
        )
        self._state["stage"] = "distillation_training"
        with self.jobs.exclusive_gpu_slot():
            train_output, training_seconds = self._run_process(
                [
                    str(training_python),
                    "-m",
                    "dahua_cup.pipeline.train_ntu120_distill",
                    "--ann-file",
                    str(annotation_path),
                    "--init-checkpoint",
                    str(base_checkpoint),
                    "--work-dir",
                    str(run_dir),
                    "--gpus",
                    str(self.config.gpus),
                    "--epochs",
                    str(self.config.epochs),
                    "--learning-rate",
                    str(self.config.learning_rate),
                ],
                environment=environment,
            )
            (run_dir / "training.log").write_text(
                train_output, encoding="utf-8"
            )
            candidates = sorted(
                run_dir.glob("best*.pth"),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates and (run_dir / "latest.pth").is_file():
                candidates = [run_dir / "latest.pth"]
            if not candidates:
                raise FileNotFoundError("training produced no candidate checkpoint")
            training_checkpoint = candidates[-1]
            weight_checkpoint = run_dir / "candidate_weights.pth"
            self._run_process(
                [
                    str(training_python),
                    "-m",
                    "dahua_cup.pipeline.export_weight_checkpoint",
                    "--input",
                    str(training_checkpoint),
                    "--output",
                    str(weight_checkpoint),
                ],
                environment=environment,
            )
            self._state["stage"] = "candidate_evaluation"
            baseline_metrics = self._evaluate_checkpoint(
                training_python,
                annotation_path,
                base_checkpoint,
                run_dir,
                "baseline",
            )
            candidate_metrics = self._evaluate_checkpoint(
                training_python,
                annotation_path,
                weight_checkpoint,
                run_dir,
                "candidate",
            )
        candidate_metrics["edge_size_bytes"] = weight_checkpoint.stat().st_size
        gate = evaluate_ntu120_release(
            candidate_metrics,
            baseline_metrics,
            candidate_size_bytes=weight_checkpoint.stat().st_size,
            maximum_checkpoint_bytes=self.config.maximum_checkpoint_bytes,
            minimum_mean_class_improvement=(
                self.config.minimum_mean_class_improvement
            ),
            maximum_top1_drop=self.config.maximum_top1_drop,
        )
        report = {
            "schema_version": "ntu120_release_gate.v1",
            **gate,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "base_checkpoint": str(base_checkpoint),
            "candidate_checkpoint": str(weight_checkpoint),
            "training_seconds": training_seconds,
            "evaluated_at": _now(),
        }
        report_path = run_dir / "release_report.json"
        _atomic_json(report_path, report)
        model_id = f"protogcn-ntu120-{run_id}"
        registry = ModelRegistry(self.registry_path)
        pointer = ProductionPointer(self.pointer_path)
        self._ensure_baseline_pointer(
            registry,
            pointer,
            base_checkpoint,
            baseline_metrics,
        )
        registry.register(
            {
                "model_id": model_id,
                "dataset_id": f"ntu120-auto-{run_id}",
                "checkpoint_path": str(weight_checkpoint),
                "config_path": str(
                    self.settings.repository_root
                    / "dahua_cup"
                    / "configs"
                    / "protogcn"
                    / "ntu120_ntu25_bone_distill.py"
                ),
                "config_hash": file_hash(
                    self.settings.repository_root
                    / "dahua_cup"
                    / "configs"
                    / "protogcn"
                    / "ntu120_ntu25_bone_distill.py"
                ),
                "checkpoint_hash": _sha256(weight_checkpoint),
                "metrics": candidate_metrics,
                "edge_size_bytes": weight_checkpoint.stat().st_size,
                "latency": {
                    "device": "not_measured",
                    "milliseconds": None,
                },
                "base_checkpoint": str(base_checkpoint),
                "release_report": str(report_path),
            }
        )
        if report["passed"]:
            registry.transition(model_id, "validated")
            registry.transition(model_id, "canary")
            registry.transition(model_id, "production")
            previous_id = pointer.read().get("current_model_id")
            pointer.promote(model_id, report)
            if previous_id:
                try:
                    registry.transition(previous_id, "archived")
                except (KeyError, ValueError):
                    pass
            outcome = "promoted"
            registered_status = "production"
        else:
            registry.transition(model_id, "archived")
            outcome = "rejected_original_checkpoint_preserved"
            registered_status = "archived"
        training_state["last_report"] = report
        training_state["attempted_fingerprints"] = list(
            training_state["last_attempt_fingerprints"]
        )
        training_state["last_model_id"] = model_id
        training_state["outcome"] = outcome
        training_state["registered_status"] = registered_status
