"""Bounded background jobs for media conversion and model inference."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dahua_cup.pipeline.common import render_command
from dahua_cup.semantic_teacher.incremental.campus_dataset import (
    CampusIncrementalDataset,
    IncrementalRound,
)
from dahua_cup.semantic_teacher.prompts.prompt_builder import PROMPT_VERSION

from .baseline import Campus6Baseline
from .config import Settings
from .gpu import GPUManager
from .hard_samples import evaluate_hard_sample
from .qwen_api import run_qwen_api
from .remote import (
    provision_remote as provision_qwen_remote,
    run_remote_incremental_training,
    run_remote_qwen,
    test_connection as test_qwen_remote_connection,
)
from .store import INTERRUPTED_JOB_MESSAGE, ReviewStore


ACTIONS = frozenset(("pose", "classify", "teacher", "full"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("sample_id cannot be converted to a safe artifact name")
    return result


def set_command_option(command: List[str], option: str, value: str) -> List[str]:
    """Return a command with one CLI option appended or forcibly replaced."""
    result = list(command)
    try:
        index = result.index(option)
    except ValueError:
        result.extend((option, value))
        return result
    if index + 1 >= len(result):
        result.append(value)
    else:
        result[index + 1] = value
    return result


def pose_quality_metrics(path: Path) -> dict:
    """Measure usable COCO-17 joint coverage without penalizing empty slots."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("pose quality calculation requires numpy") from exc
    try:
        with np.load(path, allow_pickle=False) as artifact:
            scores_source = np.asarray(artifact["keypoint_score"], dtype=np.float32) if "keypoint_score" in artifact else None
            valid = np.asarray(artifact["valid_mask"], dtype=bool) if "valid_mask" in artifact else (scores_source >= 0.20 if scores_source is not None else None)
            if valid is None:
                raise KeyError("valid_mask/keypoint_score")
            scores = (
                scores_source if scores_source is not None
                else valid.astype(np.float32)
            )
    except (OSError, KeyError, ValueError) as exc:
        return {
            "quality": 0.0,
            "pose_coverage": 0.0,
            "mean_joint_score": 0.0,
            "status": "invalid_pose_artifact",
            "reason": str(exc),
        }
    if valid.ndim != 3 or scores.shape != valid.shape or valid.shape[1] == 0:
        return {
            "quality": 0.0,
            "pose_coverage": 0.0,
            "mean_joint_score": 0.0,
            "status": "invalid_pose_shape",
            "reason": f"valid={valid.shape}, scores={scores.shape}",
        }
    if not np.isfinite(scores).all():
        return {
            "quality": 0.0,
            "pose_coverage": 0.0,
            "mean_joint_score": 0.0,
            "status": "non_finite_pose_score",
            "reason": "keypoint_score contains non-finite values",
        }
    active_people = valid.any(axis=(1, 2))
    if not active_people.any():
        return {
            "quality": 0.0,
            "pose_coverage": 0.0,
            "mean_joint_score": 0.0,
            "status": "no_pose",
            "reason": "no active pose slot",
        }
    active_valid = valid[active_people]
    active_scores = scores[active_people]
    coverage = float(active_valid.mean())
    mean_score = float(active_scores[active_valid].mean()) if active_valid.any() else 0.0
    quality = max(0.0, min(1.0, 0.70 * coverage + 0.30 * mean_score))
    return {
        "quality": quality,
        "pose_coverage": coverage,
        "mean_joint_score": max(0.0, min(1.0, mean_score)),
        "active_people": int(active_people.sum()),
        "status": "ok",
    }


def student_instability(prediction: dict) -> dict:
    """Compare the current Campus6 Top-1 with bounded prior-run summaries."""
    topk = prediction.get("topk") or []
    history = list(prediction.get("inference_history") or [])
    try:
        current_label = str(topk[0]["label"])
        current_score = float(topk[0]["score"])
    except (IndexError, KeyError, TypeError, ValueError):
        return {
            "score": 1.0,
            "label_flip": True,
            "confidence_range": 1.0,
            "runs_compared": 0,
            "reason": "missing_current_prediction",
        }
    prior = []
    for item in history[-4:]:
        try:
            prior.append((str(item["top1_label"]), float(item["top1_score"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not prior:
        return {
            "score": 0.0,
            "label_flip": False,
            "confidence_range": 0.0,
            "runs_compared": 0,
            "reason": "insufficient_history",
        }
    label_flip = any(label != current_label for label, _ in prior)
    scores = [score for _, score in prior] + [current_score]
    confidence_range = max(scores) - min(scores)
    variation = max(0.0, min(1.0, confidence_range / 0.25))
    return {
        "score": 1.0 if label_flip else variation,
        "label_flip": label_flip,
        "confidence_range": confidence_range,
        "runs_compared": len(prior),
        "reason": "label_flip" if label_flip else "confidence_variation",
    }


def teacher_gate_decision(
    prediction: dict,
    confidence_threshold: float,
    *,
    pose_quality: float = 1.0,
    pose_quality_threshold: float = 0.70,
    instability_score: float | None = None,
    instability_threshold: float = 0.60,
) -> dict:
    """Route a Campus6 result to student, Qwen, or direct human review."""
    thresholds = (
        confidence_threshold,
        pose_quality_threshold,
        instability_threshold,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("teacher gate thresholds must be in [0, 1]")
    pose_quality = max(0.0, min(1.0, float(pose_quality)))
    instability = (
        student_instability(prediction)
        if instability_score is None
        else {
            "score": max(0.0, min(1.0, float(instability_score))),
            "reason": "provided",
        }
    )
    topk = prediction.get("topk") or []
    try:
        confidence = float(topk[0]["score"])
        top1_label = str(topk[0]["label"])
    except (IndexError, KeyError, TypeError, ValueError):
        confidence = None
        top1_label = None
    try:
        top2_score = float(topk[1]["score"])
    except (IndexError, KeyError, TypeError, ValueError):
        top2_score = None
    valid_confidence = (
        confidence is not None and math.isfinite(confidence)
        and 0 <= confidence <= 1
    )
    valid_top2 = (
        top2_score is not None and math.isfinite(top2_score)
        and 0 <= top2_score <= 1
    )
    margin = (
        max(0.0, confidence - top2_score)
        if valid_confidence and valid_top2 else None
    )
    reasons = []
    if not valid_confidence:
        reasons.append("invalid_student_top1_confidence")
    elif confidence < confidence_threshold:
        reasons.append("student_confidence_below_threshold")
    if margin is None:
        reasons.append("missing_student_top2_margin")
    elif margin < confidence_threshold:
        reasons.append("student_margin_below_threshold")
    if instability["score"] >= instability_threshold:
        reasons.append("student_prediction_instability")

    pose_failure = pose_quality < pose_quality_threshold
    if pose_failure:
        # Pose quality is an uncertainty signal, not a reason to suppress
        # either model or to force human review by itself.  A structurally
        # usable low-quality pose asks the semantic teacher for a second
        # opinion; downstream disagreement/instability decides review.
        reasons.insert(0, "pose_quality_below_threshold")
    route = "call_teacher" if reasons else "use_student"
    return {
        "route": route,
        "threshold": confidence_threshold,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": confidence_threshold,
        "pose_quality_threshold": pose_quality_threshold,
        "instability_threshold": instability_threshold,
        "top1_label": top1_label,
        "student_confidence": confidence,
        "top1_top2_margin": margin,
        "pose_quality": pose_quality,
        "instability": instability,
        "triggered": route == "call_teacher",
        "hard_sample": (
            pose_failure
            or instability["score"] >= instability_threshold
        ),
        "reason": reasons[0] if reasons else "student_result_reliable",
        "reasons": reasons,
        "confidence_method": (
            "student_validation_temperature_scaled_softmax"
            if (prediction.get("confidence_calibration") or {}).get("method")
            == "temperature_scaling"
            else "student_top1_softmax_uncalibrated"
        ),
    }


def teacher_student_conflict(
    prediction: dict, teacher_payload: dict, threshold: float
) -> dict:
    """Detect a high-confidence label conflict in the Campus6 label space."""
    if not 0 <= threshold <= 1:
        raise ValueError("teacher conflict threshold must be in [0, 1]")
    topk = prediction.get("topk") or []
    result = teacher_payload.get("result") or {}
    try:
        student_label = str(topk[0]["label"])
        student_confidence = float(topk[0]["score"])
        teacher_label = str(result["label"])
        teacher_confidence = float(result["confidence"])
    except (IndexError, KeyError, TypeError, ValueError):
        return {
            "conflict": False,
            "reason": "missing_conflict_inputs",
            "threshold": threshold,
        }
    conflict = (
        student_label != teacher_label
        and student_confidence >= threshold
        and teacher_confidence >= threshold
    )
    return {
        "conflict": conflict,
        "reason": (
            "high_confidence_student_teacher_conflict"
            if conflict else "no_high_confidence_conflict"
        ),
        "threshold": threshold,
        "student_label": student_label,
        "student_confidence": student_confidence,
        "teacher_label": teacher_label,
        "teacher_confidence": teacher_confidence,
        "confidence_method": (
            "student_validation_temperature_scaled_and_teacher_self_report"
            if (prediction.get("confidence_calibration") or {}).get("method")
            == "temperature_scaling"
            else "student_softmax_and_teacher_self_report_uncalibrated"
        ),
    }


class JobManager:
    def __init__(
        self,
        settings: Settings,
        store: ReviewStore,
        baseline: Optional[Campus6Baseline] = None,
    ):
        self.settings = settings
        self.store = store
        self.baseline = baseline
        self.gpus = GPUManager(settings.gpu_settings_path)
        self.gpu_condition = threading.Condition()
        self.active_regular_gpu_jobs = 0
        self.teacher_waiting = False
        self.teacher_active = False
        self.submission_lock = threading.Lock()
        self.active_sample_jobs: Dict[str, tuple[str, str]] = {}
        self._shutdown = threading.Event()
        self._process_lock = threading.Lock()
        self._processes: Dict[int, subprocess.Popen] = {}
        self._futures: set[Future] = set()
        self._continuous_stop = threading.Event()
        self._continuous_thread: Optional[threading.Thread] = None
        # Keep JobManager usable by small test/tool Settings stubs which only
        # provide the worker-related fields.  Real Web Settings always expose
        # the explicit dataset/runtime properties.
        repository_root = Path(settings.repository_root)
        self.incremental = CampusIncrementalDataset(
            Path(getattr(settings, "dataset_root", repository_root / "dataset")),
            Path(getattr(settings, "runtime_root", repository_root / "runtime")),
            batch_size=int(getattr(settings, "incremental_batch_size", 50)),
            seed=int(getattr(settings, "incremental_seed", 20260827)),
        )
        self.incremental.ensure_layout()
        self._incremental_lock = threading.Lock()
        self._incremental_future: Optional[Future] = None
        self._next_incremental_check = 0.0
        self._continuous_status = {
            "enabled": False,
            "state": "stopped",
            "last_sample_id": "",
            "last_error": "",
        }
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="dahua-web"
        )

    def start_continuous_pipeline(self) -> None:
        """Continuously materialize pose, student, and eligible teacher results.

        This is intentionally server-owned.  The browser only observes and
        reviews completed results; it never uploads RGB or starts inference.
        """
        if os.environ.get("DAHUA_CONTINUOUS_PIPELINE", "1") != "1":
            return
        if self._continuous_thread and self._continuous_thread.is_alive():
            return
        self._continuous_stop.clear()
        self._continuous_status.update(enabled=True, state="starting", last_error="")
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop,
            name="dahua-campus6-continuous",
            daemon=True,
        )
        self._continuous_thread.start()

    def stop_continuous_pipeline(self) -> None:
        self._continuous_stop.set()
        if self._continuous_thread:
            self._continuous_thread.join(timeout=2)
        self._continuous_status["state"] = "stopped"

    def shutdown(self) -> None:
        """Stop the service-owned queue and every local worker process group."""
        self._shutdown.set()
        self.stop_continuous_pipeline()
        with self.submission_lock:
            futures = list(self._futures)
        for future in futures:
            future.cancel()
        with self._process_lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 8
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            if process.poll() is None:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self.executor.shutdown(wait=True)

    def continuous_status(self) -> dict:
        return dict(self._continuous_status)

    def model_status(self) -> dict:
        """Summarize production/candidate lifecycle without exposing paths."""
        checkpoint = self.settings.resolve_student_checkpoint()
        generated_at = (
            datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat()
            if checkpoint and checkpoint.is_file()
            else None
        )
        metrics = (
            self.baseline.evaluation_summary()
            if self.baseline is not None and self.baseline.available
            else None
        )
        registry_path = self.settings.runtime_root / "settings/model_registry.jsonl"
        pointer_path = self.settings.runtime_root / "settings/production_pointer.json"
        records = []
        if registry_path.is_file():
            for line in registry_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except (TypeError, ValueError):
                    continue
        latest = {}
        for record in records:
            if record.get("model_id"):
                latest[str(record["model_id"])] = record
        pointer = {}
        if pointer_path.is_file():
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                pointer = {}
        candidate_records = [
            value for value in latest.values()
            if value.get("status") in {"candidate", "validated", "archived"}
        ]
        candidate_records.sort(
            key=lambda value: str(
                value.get("updated_at") or value.get("created_at") or ""
            ),
            reverse=True,
        )
        previous_model_id = pointer.get("previous_model_id")
        return {
            "schema_version": "campus6_model_status.v1",
            "production": {
                "model_id": pointer.get("current_model_id") or "m1kd-int8-campus6",
                "name": "M1KD QAT INT8 + Logits KD",
                "status": "production",
                "generated_at": generated_at,
                "deployed_at": (
                    (pointer.get("history") or [{}])[-1].get("created_at")
                    if pointer.get("history") else generated_at
                ),
                "size_mb": (
                    checkpoint.stat().st_size / 1_000_000
                    if checkpoint and checkpoint.is_file() else None
                ),
                "metrics": metrics,
            },
            "candidate": candidate_records[0] if candidate_records else None,
            "lifecycle": ["candidate", "validated", "production"],
            "rollback": {
                "available": bool(previous_model_id),
                "previous_model_id": previous_model_id,
                "history_count": len(pointer.get("history") or []),
            },
        }

    def incremental_training_status(self) -> dict:
        """Read the auditable trainer heartbeat and derive a truthful ETA."""
        model = self.model_status()
        production = model.get("production") or {}
        last_training_at = production.get("generated_at")
        status_path = (
            self.settings.runtime_root / "settings/incremental_training.json"
        )
        value = {}
        if status_path.is_file():
            try:
                value = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                value = {"status": "failed", "message": "训练状态文件无法读取"}
        status = str(value.get("status") or "idle")
        if status not in {"idle", "running", "completed", "failed"}:
            status = "failed"
        running = status == "running"
        pid = value.get("pid")
        if running and pid:
            try:
                os.kill(int(pid), 0)
            except (OSError, TypeError, ValueError):
                running = False
                status = "failed"
                value["message"] = "训练进程已结束，但未写入完成状态"
        try:
            progress = max(0.0, min(1.0, float(value.get("progress", 0.0))))
        except (TypeError, ValueError):
            progress = 0.0
        remaining = value.get("estimated_remaining_seconds")
        if running and remaining is None and value.get("estimated_total_seconds"):
            try:
                started = datetime.fromisoformat(
                    str(value["started_at"]).replace("Z", "+00:00")
                )
                elapsed = max(
                    0.0,
                    (datetime.now(timezone.utc) - started).total_seconds(),
                )
                remaining = max(
                    0,
                    int(float(value["estimated_total_seconds"]) - elapsed),
                )
            except (KeyError, TypeError, ValueError):
                remaining = None
        dataset = self.incremental.status()
        return {
            "schema_version": "incremental_training_status.v1",
            "status": status,
            "running": running,
            "stage": value.get("stage") or ("waiting_for_data" if not running else "train"),
            "progress": progress,
            "started_at": value.get("started_at"),
            "updated_at": value.get("updated_at"),
            "last_training_at": last_training_at,
            "new_sample_count": dataset["staged_sample_count"],
            "incremental_dataset": dataset,
            "estimated_remaining_seconds": remaining if running else None,
            "message": value.get("message") or "",
        }

    @property
    def _incremental_status_path(self) -> Path:
        return self.settings.runtime_root / "settings" / "incremental_training.json"

    def _write_incremental_status(self, **value: object) -> None:
        payload = {"schema_version": "incremental_training_status.v1", **value}
        path = self._incremental_status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def enroll_reviewed_hard_sample(self, sample_id: str) -> dict:
        """Materialize a human-confirmed hard example in campus_increment.

        The review itself remains valid when the feature is not available yet:
        a later call (or a retry after inference) can stage it without asking
        the reviewer to relabel the sample.
        """
        sample = self.store.get_sample(sample_id)
        if not sample.get("incremental_pool") or sample.get("status") != "reviewed":
            return {"staged": False, "reason": "not_human_confirmed_hard_sample", **self.incremental.status()}
        paths = self.artifacts(sample_id)
        try:
            result = self.incremental.stage(
                sample,
                feature_path=paths["feature"],
            )
        except FileNotFoundError:
            return {"staged": False, "reason": "waiting_for_pose_artifact", **self.incremental.status()}
        # Upload/training is deliberately scheduled by the one-minute monitor,
        # so a browser review never blocks on a large SSH transfer.
        return result

    def _start_incremental_training_if_ready(self) -> dict:
        """Queue one scheduled upload at most once, without occupying local GPUs."""
        with self._incremental_lock:
            if self._incremental_future and not self._incremental_future.done():
                return {"queued": False, "reason": "already_running"}
            state = self.incremental.status()
            if not state["ready"]:
                return {"queued": False, "reason": "batch_not_ready"}
            self._incremental_future = self.executor.submit(self._run_incremental_training)
            self._futures.add(self._incremental_future)
            self._incremental_future.add_done_callback(self._futures.discard)
            return {"queued": True, "reason": "batch_ready"}

    def _run_incremental_training(self) -> None:
        round_: Optional[IncrementalRound] = None
        try:
            # Freeze exactly one documented 50-sample batch.  Reviews that
            # arrive while it trains remain in campus_increment for the next
            # transaction.
            round_ = self.incremental.prepare_round()
            work_dir = round_.root / "work_dir"
            self._write_incremental_status(
                status="running", stage="upload", progress=0.10,
                started_at=_now(), updated_at=_now(), round_id=round_.round_id,
                sample_count=len(round_.sample_ids),
                message="{} 条人工确认难例已冻结，并与最多 250 条历史训练样本组成重放集，正在上传服务器".format(
                    len(round_.sample_ids)
                ),
            )
            remote = getattr(self.settings, "incremental_remote", lambda: {})()
            if remote.get("enabled"):
                remote_result = run_remote_incremental_training(
                    remote,
                    round_id=round_.round_id,
                    annotation=round_.training_annotation,
                    remote_python=self.settings.incremental_remote_python,
                    init_checkpoint=self.settings.incremental_remote_init_checkpoint,
                    epochs=10,
                    gpu_id=self.settings.incremental_remote_gpu_id,
                )
                training_metadata = {"execution": "remote_ssh", **remote_result}
            else:
                command_template = self.settings.default_incremental_train_command()
                if not command_template:
                    raise RuntimeError(
                        "未配置可用的远程训练服务器，也未配置本地 DAHUA_STUDENT_PYTHON/训练基线"
                    )
                values = {
                    "ann_file": shlex.quote(str(round_.training_annotation)),
                    "work_dir": shlex.quote(str(work_dir)),
                    "init_checkpoint": shlex.quote(str(self.settings.training_baseline_checkpoint() or "")),
                }
                command = render_command(command_template, **values)
                with self.exclusive_gpu_slot():
                    log = self._execute(command)
                training_metadata = {
                    "execution": "local",
                    "command": command,
                    "log_tail": log[-4000:],
                }
            self._write_incremental_status(
                status="running", stage="merge", progress=0.90,
                started_at=_now(), updated_at=_now(), round_id=round_.round_id,
                sample_count=len(round_.sample_ids),
                message="训练已完成，正在合并 campus_all",
            )
            merged = self.incremental.complete_round(
                round_, training_metadata=training_metadata
            )
            self._write_incremental_status(
                status="completed", stage="completed", progress=1.0,
                started_at=_now(), updated_at=_now(), round_id=round_.round_id,
                sample_count=len(round_.sample_ids), message="增量训练成功；已合并 campus_all，本批 campus_increment 已清理",
                merge=merged,
            )
        except Exception as exc:
            if round_ is not None:
                self.incremental.fail_round(round_, str(exc))
            self._write_incremental_status(
                status="failed", stage="failed", progress=0.0,
                updated_at=_now(), round_id=round_.round_id if round_ else None,
                message=str(exc)[:1000],
            )

    def _auto_retry_allowed(self, sample_id: str, action: str) -> bool:
        """Gate automatic resubmission of failed jobs in the continuous loop.

        Jobs interrupted by a Web restart carry the fixed message
        ``INTERRUPTED_JOB_MESSAGE`` and may be retried without limit; they
        failed for environmental reasons, not because of the sample.  Other
        failures consume a small per-sample budget so a deterministically
        failing sample cannot occupy the queue forever.
        """
        jobs = self.store.get_sample(sample_id).get("jobs", [])
        failures = [
            job for job in jobs
            if job.get("action") == action and job.get("status") == "failed"
        ]
        model_failures = [
            job for job in failures
            if (job.get("message") or "") != INTERRUPTED_JOB_MESSAGE
        ]
        return len(model_failures) < 3

    def _continuous_loop(self) -> None:
        self._continuous_status["state"] = "running"
        while not self._continuous_stop.is_set():
            try:
                self._poll_incremental_upload()
                batch = self.store.list_samples(status="pending", limit=200)["items"]
                submitted = False
                for item in batch:
                    sample_id = item["sample_id"]
                    with self.submission_lock:
                        if sample_id in self.active_sample_jobs:
                            continue
                    prediction = self.prediction(sample_id)
                    teacher = self.teacher_state(sample_id) if prediction else {}
                    if prediction is None:
                        # Initial Campus6 records are already standardized
                        # COCO-17 skeletons.  Do not feed their rendered pose
                        # video back into RTMPose; only run the student.
                        action = (
                            "classify"
                            if self.artifacts(sample_id)["feature"].is_file()
                            else "full"
                        )
                    elif (
                        (prediction.get("teacher_gate") or {}).get("triggered")
                        and (
                            teacher.get("status") in {"not_run", "not_enabled"}
                            or (
                                teacher.get("status") == "failed"
                                and self._auto_retry_allowed(
                                    sample_id, "teacher"
                                )
                            )
                        )
                        and self.capability_state()["qwen_teacher"]["enabled"]
                    ):
                        action = "teacher"
                    else:
                        continue
                    latest = self.store.get_sample(sample_id).get("jobs", [])
                    if (
                        latest
                        and latest[0].get("action") == action
                        and latest[0].get("status") == "failed"
                        and not self._auto_retry_allowed(sample_id, action)
                    ):
                        continue
                    self.submit(sample_id, action)
                    self._continuous_status["last_sample_id"] = sample_id
                    self._continuous_status["last_error"] = ""
                    submitted = True
                    break
                with self.gpu_condition:
                    teacher_busy = self.teacher_active or self.teacher_waiting
                if (
                    not submitted
                    and self.baseline is not None
                    and not teacher_busy
                ):
                    capability = self.capability_state()["qwen_teacher"]
                    if capability["enabled"]:
                        for item in self.hard_samples(limit=50)["items"]:
                            teacher_status = item.get("teacher", {}).get("status")
                            sample_id = item["sample_id"]
                            if teacher_status != "not_run":
                                retryable = (
                                    teacher_status == "failed"
                                    and self._auto_retry_allowed(
                                        sample_id, "teacher"
                                    )
                                )
                                if not retryable:
                                    continue
                            with self.submission_lock:
                                if sample_id in self.active_sample_jobs:
                                    continue
                            self.submit(sample_id, "teacher")
                            self._continuous_status["last_sample_id"] = sample_id
                            self._continuous_status["last_error"] = ""
                            submitted = True
                            break
                self._continuous_stop.wait(2 if submitted else 20)
            except Exception as exc:
                self._continuous_status["last_error"] = f"{type(exc).__name__}: {exc}"
                self._continuous_stop.wait(20)

    def _poll_incremental_upload(self) -> None:
        """Every minute, start one remote/local incremental job when >=50 await.

        The check only queues a background future; it never takes an inference
        GPU slot or waits for SSH, so normal local video jobs keep running.
        """
        now = time.monotonic()
        if now < self._next_incremental_check:
            return
        self._next_incremental_check = now + 60.0
        state = self.incremental.status()
        if state["staged_sample_count"] < self.incremental.batch_size:
            return
        queued = self._start_incremental_training_if_ready()
        if queued.get("queued"):
            self._write_incremental_status(
                status="queued", stage="upload", progress=0.0,
                updated_at=_now(), sample_count=state["staged_sample_count"],
                message="每分钟检测到 {} 条人工确认难例，已排队冻结、划分并上传".format(
                    state["staged_sample_count"]
                ),
            )

    @contextmanager
    def regular_gpu_slot(self):
        """Allow normal GPU jobs concurrently, but never alongside Qwen8B."""
        with self.gpu_condition:
            while self.teacher_active or self.teacher_waiting:
                self.gpu_condition.wait()
            self.active_regular_gpu_jobs += 1
        try:
            yield
        finally:
            with self.gpu_condition:
                self.active_regular_gpu_jobs -= 1
                self.gpu_condition.notify_all()

    @contextmanager
    def teacher_gpu_slot(self):
        """Give the multi-GPU teacher exclusive access to visible GPUs."""
        with self.gpu_condition:
            self.teacher_waiting = True
            try:
                while self.teacher_active or self.active_regular_gpu_jobs:
                    self.gpu_condition.wait()
                self.teacher_active = True
            finally:
                self.teacher_waiting = False
        try:
            yield
        finally:
            with self.gpu_condition:
                self.teacher_active = False
                self.gpu_condition.notify_all()

    @contextmanager
    def exclusive_gpu_slot(self):
        """Reserve all inference GPUs for Qwen or incremental training."""
        with self.teacher_gpu_slot():
            yield

    def artifacts(self, sample_id: str) -> Dict[str, Path]:
        name = safe_name(sample_id)
        root = self.settings.artifact_root
        return {
            "feature": root / "features" / f"{name}.npz",
            "pose_video": root / "pose_videos" / f"{name}.mp4",
            "prediction": root / "predictions" / f"{name}.json",
            "teacher": root / "teachers" / f"{name}.json",
            "work_dir": root / "work" / name,
        }

    def capability_state(self) -> dict:
        pose_command = self.settings.default_pose_command()
        pose_backend = self.settings.pose_backend_status()
        student_command = self.settings.default_student_command()
        teacher_command = self.settings.default_teacher_command()
        remote_teacher = self.settings.qwen_remote()
        remote_enabled = bool(remote_teacher["enabled"])
        qwen_api_status = getattr(
            self.settings,
            "qwen_api_status",
            lambda: {"enabled": False, "available": False, "reason": "未配置 Qwen API 备用通道"},
        )()
        api_enabled = bool(qwen_api_status.get("available"))
        # A configured remote Qwen host does not need (and must not block on)
        # a local nvidia-smi probe.  This matters for CPU-only workstations.
        teacher_gpus = (
            {
                "enabled": True,
                "gpu_ids": [],
                "required_gpu_count": 0,
                "minimum_free_memory_mb": 0,
                "maximum_utilization_percent": 0,
                "idle_gpu_ids": [],
                "selection_mode": "remote",
                "permitted_gpu_ids": [],
            }
            if remote_enabled else self.gpus.teacher_availability()
        )
        return {
            "pose_extraction": {
                "enabled": bool(pose_command),
                "reason": (
                    "" if pose_command else
                    "已选择的 TensorRT 骨架模型尚未导出或验证"
                    if pose_backend["selected"] in {"tensorrt_fp16", "tensorrt_int8"} else
                    "尚未配置 RTMPose Python 环境或 DAHUA_VIS_POSE_COMMAND"
                ),
                "backend": pose_backend["selected"],
            },
            "campus6_inference": {
                "enabled": bool(student_command),
                "reason": "" if student_command else "尚未配置 Campus6 ProtoGCN 权重或 Python 环境",
            },
            "manual_review": {"enabled": True, "reason": ""},
            "dataset_export": {"enabled": True, "reason": ""},
            "qwen_teacher": {
                "enabled": (
                    remote_enabled
                    or (bool(teacher_command) and teacher_gpus["enabled"])
                    or api_enabled
                ),
                "reason": (
                    "" if (
                        remote_enabled
                        or (teacher_command and teacher_gpus["enabled"])
                        or api_enabled
                    )
                    else (
                        "尚未配置服务器 Qwen 模型/教师环境，且 Qwen API 备用通道不可用"
                        if not teacher_command else
                        "Qwen3-VL-8B 需要 {} 张空闲 GPU（每张至少 {} MiB、利用率不高于 {}%）；当前可用：{}"
                        .format(
                            teacher_gpus["required_gpu_count"],
                            teacher_gpus["minimum_free_memory_mb"],
                            teacher_gpus["maximum_utilization_percent"],
                            teacher_gpus["idle_gpu_ids"] or "无",
                        )
                    )
                ),
                "gpu_admission": teacher_gpus,
                "execution": (
                    "remote_ssh_then_qwen_api" if remote_enabled and api_enabled
                    else "remote_ssh" if remote_enabled
                    else "local_then_qwen_api" if teacher_command and api_enabled
                    else "local" if teacher_command
                    else "qwen_api"
                ),
                "api_fallback": {
                    "enabled": bool(qwen_api_status.get("enabled")),
                    "available": api_enabled,
                    "reason": str(qwen_api_status.get("reason") or ""),
                    "model": str(qwen_api_status.get("model") or ""),
                },
            },
            "live_camera": {"enabled": False, "reason": "第一阶段只处理服务器文件"},
        }

    def local_inference_capability(self) -> dict:
        """Cheap preflight for a local video upload.

        It intentionally does not probe Qwen GPU admission.  The upload only
        needs the local pose and GCN worker commands; Qwen remains governed by
        its later uncertainty gate.  This prevents a browser upload from
        waiting for several ``nvidia-smi`` timeouts on a CPU-only notebook.
        """
        pose_command = self.settings.default_pose_command()
        student_command = self.settings.default_student_command()
        pose_backend = self.settings.pose_backend_status()
        return {
            "pose_extraction": {
                "enabled": bool(pose_command),
                "reason": (
                    "" if pose_command else
                    "已选择的 TensorRT 骨架模型尚未导出或验证"
                    if pose_backend["selected"] in {"tensorrt_fp16", "tensorrt_int8"} else
                    "尚未配置 RTMPose Python 环境或 DAHUA_VIS_POSE_COMMAND"
                ),
            },
            "campus6_inference": {
                "enabled": bool(student_command),
                "reason": "" if student_command else "尚未配置 Campus6 ProtoGCN 权重或 Python 环境",
            },
        }

    def test_qwen_remote_connection(self) -> None:
        test_qwen_remote_connection(self.settings.qwen_remote())

    def provision_qwen_remote(self) -> dict:
        return provision_qwen_remote(self.settings.qwen_remote())

    def submit(self, sample_id: str, action: str) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        capabilities = self.capability_state()
        required = {
            "pose": ("pose_extraction",),
            "classify": ("campus6_inference",),
            "teacher": ("qwen_teacher",),
            "full": ("pose_extraction", "campus6_inference"),
        }[action]
        unavailable = [capabilities[name]["reason"] for name in required if not capabilities[name]["enabled"]]
        if unavailable:
            raise ValueError("；".join(unavailable))
        if action == "teacher":
            self.materialize_baseline_inputs(sample_id)
            paths = self.artifacts(sample_id)
            if not paths["feature"].is_file() or not paths["pose_video"].is_file():
                raise ValueError("请先运行骨架提取并生成骨架视频")
        with self.submission_lock:
            active = self.active_sample_jobs.get(sample_id)
            if active is not None:
                active_job_id, active_action = active
                try:
                    active_job = self.store.get_job(active_job_id)
                except KeyError:
                    self.active_sample_jobs.pop(sample_id, None)
                else:
                    if active_job["status"] in {"queued", "running"}:
                        if active_action != action:
                            raise ValueError(
                                "样本已有 {} 任务正在运行，请等待完成后再执行 {}"
                                .format(active_action, action)
                            )
                        active_job["reused"] = True
                        return active_job
                    self.active_sample_jobs.pop(sample_id, None)

            pose_device = ""
            student_device = ""
            if action in {"pose", "full"}:
                pose_device = self._acquire_execution_device("pose")
            if action in {"classify", "full"}:
                student_device = self._acquire_execution_device("student")
            job = self.store.create_job(
                sample_id,
                action,
                pose_device=pose_device,
                student_device=student_device,
            )
            self.active_sample_jobs[sample_id] = (job["job_id"], action)
            try:
                future = self.executor.submit(self._run, job["job_id"])
                self._futures.add(future)
                future.add_done_callback(self._futures.discard)
            except Exception:
                self.active_sample_jobs.pop(sample_id, None)
                if pose_device.startswith("cuda:"):
                    self.gpus.release_device("pose", int(pose_device.split(":", 1)[1]))
                if student_device.startswith("cuda:"):
                    self.gpus.release_device("student", int(student_device.split(":", 1)[1]))
                self.store.update_job(
                    job["job_id"],
                    status="failed",
                    message="后台任务提交失败",
                    finished_at=_now(),
                )
                raise
            job["reused"] = False
            return job

    def _acquire_execution_device(self, kind: str) -> str:
        """Choose CUDA when usable, with an explicit local auto CPU fallback."""
        mode = getattr(self.settings, "local_inference_device", "server")
        if mode == "cpu":
            return "cpu"
        try:
            return "cuda:{}".format(self.gpus.acquire_device(kind))
        except (OSError, RuntimeError, ValueError):
            # ``auto`` is only used by the local launcher.  Server and forced
            # GPU modes fail visibly rather than silently changing deployment.
            if mode == "auto":
                return "cpu"
            raise

    def _release_active_job(self, sample_id: str, job_id: str) -> None:
        with self.submission_lock:
            active = self.active_sample_jobs.get(sample_id)
            if active is not None and active[0] == job_id:
                self.active_sample_jobs.pop(sample_id, None)

    def _execute(self, command: List[str], extra_env: Optional[dict] = None) -> str:
        if self._shutdown.is_set():
            raise RuntimeError(INTERRUPTED_JOB_MESSAGE)
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        try:
            process = subprocess.Popen(
                command,
                cwd=self.settings.repository_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError("无法启动后台推理任务：{}".format(exc)) from exc
        with self._process_lock:
            self._processes[process.pid] = process
        try:
            stdout, _ = process.communicate()
        finally:
            with self._process_lock:
                self._processes.pop(process.pid, None)
        if process.returncode:
            if self._shutdown.is_set():
                raise RuntimeError(INTERRUPTED_JOB_MESSAGE)
            is_teacher = any(
                "qwen_teacher_worker" in str(part) for part in command
            )
            label = "Qwen 教师分析失败" if is_teacher else "后台推理任务失败"
            raise RuntimeError(
                "{}（退出码 {}），请稍后重试".format(label, process.returncode)
            )
        return (stdout or "")[-12000:]

    def _run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if self._shutdown.is_set():
            self.store.update_job(
                job_id, status="failed", message=INTERRUPTED_JOB_MESSAGE,
                finished_at=_now(),
            )
            self._release_active_job(job["sample_id"], job_id)
            return
        sample = self.store.get_sample(job["sample_id"])
        paths = self.artifacts(job["sample_id"])
        paths["work_dir"].mkdir(parents=True, exist_ok=True)
        action = job["action"]
        logs: List[str] = []
        completion_message = "处理完成"
        pose_blocked = False
        # The local launcher resolves auto/GPU/CPU before startup.  Server
        # launches retain scheduler-managed CUDA routing.
        pose_on_cpu = job.get("pose_device") == "cpu"
        student_on_cpu = job.get("student_device") == "cpu"
        self.store.update_job(
            job_id, status="running", progress=0.02,
            message="任务已启动", started_at=_now(),
        )
        try:
            if action == "teacher":
                # Qwen receives only the derived Skeleton video.
                video = paths["pose_video"]
                if not video.is_file():
                    raise FileNotFoundError(f"骨架视频不存在：{video}")
            else:
                video = Path(sample["video_path"])
                if not video.is_file():
                    raise FileNotFoundError(f"视频不存在：{video}")
                if not self.settings.video_path_is_allowed(video):
                    raise PermissionError(
                        "视频必须位于服务器候选数据或 Web 上传目录内"
                    )
            if action in {"pose", "full"}:
                pose_execution_device = "cpu" if pose_on_cpu else "cuda:0"
                pose_location = (
                    "CPU" if pose_on_cpu else "GPU {}".format(
                        job["pose_device"].split(":", 1)[1]
                    )
                )
                self.store.update_job(
                    job_id,
                    progress=0.30,
                    message="正在 {} 提取 RTMDet/RTMPose COCO-17 骨架".format(
                        pose_location
                    ),
                )
                values = {
                    key: shlex.quote(str(value))
                    for key, value in {
                        "sample_id": job["sample_id"],
                        "video": video,
                        **paths,
                        "repository_root": self.settings.repository_root,
                        "device": pose_execution_device,
                        "physical_device": (
                            "" if pose_on_cpu else
                            int(job["pose_device"].split(":", 1)[1])
                        ),
                        "delegate": "cpu" if pose_on_cpu else "cuda",
                    }.items()
                }
                pose_command = render_command(
                    self.settings.default_pose_command(), **values
                )
                pose_command = set_command_option(
                    pose_command, "--device", pose_execution_device
                )
                pose_command = set_command_option(pose_command, "--joint-score-threshold", str(self.settings.joint_score_threshold))
                if pose_on_cpu:
                    logs.append(self._execute(pose_command))
                else:
                    pose_gpu_id = int(job["pose_device"].split(":", 1)[1])
                    with self.regular_gpu_slot():
                        logs.append(self._execute(
                            pose_command,
                            self.gpus.process_environment(pose_gpu_id),
                        ))
                self.store.update_job(job_id, progress=0.62, message="正在渲染 RTMPose17 骨架视频")
                command = [
                    sys.executable, "-m", "dahua_cup.pipeline.render_rtmpose17_pose",
                    "--feature", str(paths["feature"]),
                    "--output", str(paths["pose_video"]),
                    "--ffmpeg", str(self.settings.ffmpeg),
                    "--codec", self.settings.preview_codec,
                    "--preset", self.settings.preview_preset,
                    "--bitrate", self.settings.preview_bitrate,
                ]
                logs.append(self._execute(command))
                # The local browser upload is needed only until RTMPose has
                # produced the pose feature and skeleton-only preview.  Keep
                # no RGB copy after that successful conversion.
                if sample.get("source_dataset") == "local_upload":
                    raw_root = (self.settings.runtime_root / "videos").resolve()
                    try:
                        raw_video = Path(sample["video_path"]).resolve()
                        raw_video.relative_to(raw_root)
                        raw_video.unlink(missing_ok=True)
                        logs.append(
                            "raw_video_retention=0: removed local RGB upload "
                            "after pose extraction"
                        )
                    except (KeyError, OSError, ValueError):
                        pass
                if action == "full":
                    pose_metrics = pose_quality_metrics(paths["feature"])
                    if pose_metrics.get("status") != "ok":
                        pose_blocked = True
                        gate = teacher_gate_decision(
                            {},
                            self.settings.teacher_trigger_confidence,
                            pose_quality=pose_metrics["quality"],
                            pose_quality_threshold=(
                                self.settings.pose_quality_threshold
                            ),
                            instability_score=0.0,
                            instability_threshold=(
                                self.settings.student_instability_threshold
                            ),
                        )
                        gate["pose_metrics"] = pose_metrics
                        gate.update(
                            {
                                "route": "human_review_unusable_pose",
                                "triggered": False,
                                "hard_sample": True,
                                "reason": "pose_artifact_unusable",
                                "reasons": [
                                    "pose_artifact_unusable",
                                    *[
                                        reason
                                        for reason in gate["reasons"]
                                        if reason
                                        != "pose_quality_below_threshold"
                                    ],
                                ],
                            }
                        )
                        gate["teacher_available"] = bool(
                            self.capability_state()["qwen_teacher"]["enabled"]
                        )
                        gate["teacher_called"] = False
                        blocked_prediction = {
                            "schema_version": "protogcn_prediction.v1",
                            "status": "blocked_quality",
                            "sample_id": job["sample_id"],
                            "task": "campus6_rtmpose17",
                            "label_space_size": 6,
                            "modality": "joint",
                            "generated_at": _now(),
                            "feature": str(paths["feature"]),
                            "checkpoint": str(
                                self.settings.resolve_student_checkpoint()
                                or ""
                            ),
                            "topk": [],
                            "warning": (
                                "ProtoGCN and Qwen were skipped because the "
                                "pose artifact failed the quality gate."
                            ),
                        }
                        self._record_teacher_gate(
                            paths["prediction"], blocked_prediction, gate
                        )
                        self.store.escalate_hard_sample(
                            job["sample_id"],
                            hard_score=max(
                                0.01, 1.0 - pose_metrics["quality"]
                            ),
                            reason="pose_quality_below_threshold",
                            payload={"teacher_gate": gate},
                        )
                        completion_message = (
                            "骨架产物不可用（{}），无法运行 ProtoGCN/Qwen，"
                            "已进入人工复核"
                        ).format(pose_metrics.get("status", "unknown"))
                        logs.append(
                            "teacher_gate="
                            + json.dumps(
                                gate, ensure_ascii=False, sort_keys=True
                            )
                        )
            if action in {"classify", "full"} and not pose_blocked:
                if not paths["feature"].is_file():
                    raise FileNotFoundError("尚无骨架特征，请先运行骨架提取")
                student_execution_device = "cpu" if student_on_cpu else "cuda:0"
                student_location = (
                    "CPU" if student_on_cpu else "GPU {}".format(
                        job["student_device"].split(":", 1)[1]
                    )
                )
                self.store.update_job(
                    job_id,
                    progress=0.72,
                    message="正在 {} 运行 ProtoGCN 分类".format(student_location),
                )
                values = {
                    key: shlex.quote(str(value))
                    for key, value in {
                        "sample_id": job["sample_id"],
                        "video": video,
                        **paths,
                        "repository_root": self.settings.repository_root,
                        "device": student_execution_device,
                        "physical_device": (
                            "" if student_on_cpu else
                            int(job["student_device"].split(":", 1)[1])
                        ),
                        "student_temperature": (
                            self.settings.student_probability_temperature()
                        ),
                    }.items()
                }
                student_command = render_command(
                    self.settings.default_student_command(), **values
                )
                student_command = set_command_option(
                    student_command, "--device", student_execution_device
                )
                student_command = set_command_option(
                    student_command,
                    "--temperature",
                    str(self.settings.student_probability_temperature()),
                )
                active_checkpoint = self.settings.resolve_student_checkpoint()
                if active_checkpoint is not None:
                    student_command = set_command_option(
                        student_command,
                        "--checkpoint",
                        str(active_checkpoint),
                    )
                if student_on_cpu:
                    logs.append(self._execute(student_command))
                else:
                    student_gpu_id = int(job["student_device"].split(":", 1)[1])
                    with self.regular_gpu_slot():
                        logs.append(self._execute(
                            student_command,
                            self.gpus.process_environment(student_gpu_id),
                        ))
            if action == "full" and not pose_blocked:
                prediction = self.prediction(job["sample_id"]) or {}
                pose_metrics = pose_quality_metrics(paths["feature"])
                gate = teacher_gate_decision(
                    prediction,
                    self.settings.teacher_trigger_confidence,
                    pose_quality=pose_metrics["quality"],
                    pose_quality_threshold=self.settings.pose_quality_threshold,
                    instability_threshold=(
                        self.settings.student_instability_threshold
                    ),
                )
                gate = self._boolean_teacher_gate(prediction, gate)
                gate["pose_metrics"] = pose_metrics
                gate["confidence_method"] = (
                    "validation_temperature_scaled_int8_softmax"
                )
                teacher_command_available = bool(
                    self.capability_state()["qwen_teacher"]["enabled"]
                )
                gate["teacher_available"] = teacher_command_available
                gate["teacher_called"] = bool(
                    gate["triggered"] and teacher_command_available
                )
                self._record_teacher_gate(paths["prediction"], prediction, gate)
                logs.append(
                    "teacher_gate=" + json.dumps(
                        gate, ensure_ascii=False, sort_keys=True
                    )
                )
                if (
                    gate["instability"]["score"]
                    >= gate["instability_threshold"]
                ):
                    self.store.escalate_hard_sample(
                        job["sample_id"],
                        hard_score=gate["instability"]["score"],
                        reason="student_prediction_instability",
                        payload={"teacher_gate": gate},
                    )
                if gate["teacher_called"]:
                    self.store.update_job(
                        job_id,
                        progress=0.86,
                        message=(
                            "样本触发语义补充判断（{}），正在调用 Qwen"
                        ).format(
                            "、".join(gate["reasons"])
                        ),
                    )
                    logs.append(
                        self._execute_teacher(
                            job["sample_id"], video, paths
                        )
                    )
                    teacher_payload = json.loads(
                        paths["teacher"].read_text(encoding="utf-8")
                    )
                    conflict = teacher_student_conflict(
                        prediction,
                        teacher_payload,
                        self.settings.teacher_conflict_confidence,
                    )
                    gate["teacher_conflict"] = conflict
                    self._record_teacher_gate(
                        paths["prediction"], prediction, gate
                    )
                    logs.append(
                        "teacher_conflict=" + json.dumps(
                            conflict, ensure_ascii=False, sort_keys=True
                        )
                    )
                    teacher_result = teacher_payload.get("result") or {}
                    if conflict["conflict"]:
                        self.store.escalate_hard_sample(
                            job["sample_id"],
                            hard_score=1.0,
                            reason=conflict["reason"],
                            payload={
                                "teacher_gate": gate,
                                "teacher_conflict": conflict,
                            },
                        )
                    elif teacher_result.get("needs_review"):
                        self.store.escalate_hard_sample(
                            job["sample_id"],
                            hard_score=0.85,
                            reason="teacher_requested_review",
                            payload={"teacher_gate": gate},
                        )
                    else:
                        fused, selected_teacher = self._fuse_teacher_prediction(
                            prediction,
                            teacher_payload,
                            conflict,
                            minimum_confidence=(
                                self.settings.teacher_conflict_confidence
                            ),
                        )
                        if selected_teacher:
                            prediction = fused
                            self._record_teacher_gate(
                                paths["prediction"], prediction, gate
                            )
                            logs.append(
                                "teacher_fusion=teacher_selected_for_uncertain_student"
                            )
                elif gate["triggered"]:
                    self.store.escalate_hard_sample(
                        job["sample_id"],
                        hard_score=0.75,
                        reason="uncertain_student_teacher_unavailable",
                        payload={"teacher_gate": gate},
                    )
                    self.store.update_job(
                        job_id,
                        progress=0.94,
                        message="学生结果不确定，但 Qwen 当前未启用",
                    )
                elif gate["route"] == "use_student":
                    self.store.update_job(
                        job_id,
                        progress=0.94,
                        message=(
                            "学生置信度 {:.3f}、间隔 {:.3f} 均达到阈值，"
                            "结果稳定，跳过 Qwen"
                        ).format(
                            gate["student_confidence"],
                            gate["top1_top2_margin"],
                        ),
                    )
            if action == "teacher":
                if not paths["feature"].is_file() or not paths["pose_video"].is_file():
                    raise FileNotFoundError("尚无骨架特征或骨架视频，请先运行骨架提取")
                prediction = self.prediction(job["sample_id"]) or {}
                decision = evaluate_hard_sample(
                    prediction,
                    None,
                    confidence_threshold=(
                        self.settings.teacher_trigger_confidence
                    ),
                    conflict_confidence_threshold=(
                        self.settings.teacher_conflict_confidence
                    ),
                )
                if not decision["is_hard"]:
                    self.store.update_job(
                        job_id,
                        progress=0.94,
                        message="学生结果未通过 0.30 难例门控，已跳过 Qwen",
                    )
                    logs.append(
                        "teacher_skipped=" + json.dumps(
                            decision, ensure_ascii=False, sort_keys=True
                        )
                    )
                else:
                    self.store.update_job(
                        job_id,
                        progress=0.25,
                        message="正在运行 Qwen3-VL 大模型分析",
                    )
                    logs.append(
                        self._execute_teacher(job["sample_id"], video, paths)
                    )
                    gate = dict(prediction.get("teacher_gate") or {})
                    gate.update(
                        {
                            "teacher_called": True,
                            "teacher_available": True,
                            "reason": "student_uncertainty_gate",
                        }
                    )
                    self._record_teacher_gate(
                        paths["prediction"], prediction, gate
                    )
            self.store.update_job(
                job_id, status="completed", progress=1.0,
                message=completion_message,
                log_text="\n".join(logs)[-20000:],
                finished_at=_now(),
            )
        except Exception as exc:
            logs.append(f"{type(exc).__name__}: {exc}")
            self.store.update_job(
                job_id, status="failed", message=str(exc),
                log_text="\n".join(logs)[-20000:], finished_at=_now(),
            )
        finally:
            if job.get("pose_device", "").startswith("cuda:"):
                self.gpus.release_device(
                    "pose", int(job["pose_device"].split(":", 1)[1])
                )
            if job.get("student_device", "").startswith("cuda:"):
                self.gpus.release_device(
                    "student", int(job["student_device"].split(":", 1)[1])
                )
            self._release_active_job(job["sample_id"], job_id)

    @staticmethod
    def _record_teacher_gate(
        path: Path, prediction: dict, gate: dict
    ) -> None:
        value = dict(prediction)
        value["teacher_gate"] = gate
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _boolean_teacher_gate(self, prediction: dict, gate: dict) -> dict:
        """Route Qwen only after the mandatory Top-1 confidence gate."""
        from dahua_cup.backend.hard_samples import evaluate_hard_sample

        value = dict(prediction)
        value["teacher_gate"] = gate
        decision = evaluate_hard_sample(
            value,
            None,
            confidence_threshold=self.settings.teacher_trigger_confidence,
            conflict_confidence_threshold=self.settings.teacher_conflict_confidence,
        )
        # The hard-sample contract deliberately has an outer eligibility gate:
        # Qwen may run only when P(student Top-1) <= 0.30.  Conditions C1/C5
        # then decide whether this eligible sample actually enters the teacher
        # path; high-confidence samples must never be escalated by a secondary
        # uncertainty signal alone.
        trigger_codes = [
            code for code in decision["matched_conditions"]
            if code in {"C1", "C5"}
        ]
        gate = dict(gate)
        gate.update({
            "triggered": bool(trigger_codes),
            "hard_sample": bool(trigger_codes),
            "route": "call_teacher" if trigger_codes else "use_student",
            "reasons": trigger_codes,
            "hard_decision": decision,
        })
        return gate

    @staticmethod
    def _fuse_teacher_prediction(
        prediction: dict,
        teacher_payload: dict,
        conflict: dict,
        *,
        minimum_confidence: float,
    ) -> tuple[dict, bool]:
        """Use a clear teacher result when the uncertain student agrees enough.

        A high-confidence disagreement remains a human-review case.  The raw
        student Top-K is retained for audit; ``topk`` becomes the final result
        consumed by the existing Web view and downstream queue.
        """
        result = dict(teacher_payload.get("result") or {})
        try:
            confidence = float(result["confidence"])
            distribution = {
                str(label): float(score)
                for label, score in dict(result["distribution"]).items()
            }
        except (KeyError, TypeError, ValueError):
            return prediction, False
        if (
            conflict.get("conflict")
            or result.get("needs_review")
            or not math.isfinite(confidence)
            or confidence < minimum_confidence
            or not distribution
        ):
            return prediction, False
        fused = dict(prediction)
        fused["student_topk"] = list(prediction.get("topk") or [])
        fused["topk"] = [
            {"label": label, "score": score}
            for label, score in sorted(
                distribution.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        fused["fusion"] = {
            "schema_version": "campus6_teacher_fusion.v1",
            "selected": "teacher",
            "reason": "student_uncertain_teacher_clear",
            "teacher_confidence": confidence,
            "student_top1": (prediction.get("topk") or [{}])[0],
        }
        return fused, True

    def _execute_teacher(
        self, sample_id: str, video: Path, paths: Dict[str, Path]
    ) -> str:
        if not paths["feature"].is_file() or not paths["pose_video"].is_file():
            raise FileNotFoundError(
                "尚无骨架特征或骨架视频，请先运行骨架提取"
            )
        failures = []
        remote_teacher = self.settings.qwen_remote()
        if remote_teacher["enabled"]:
            try:
                run_remote_qwen(remote_teacher, sample_id, paths, self._execute)
                self._validate_teacher_artifact(paths["teacher"])
                return "Qwen 云端教师分析完成"
            except Exception as exc:
                # The hosted fallback is intentionally a resilience path.  It
                # is tried only after the preferred self-hosted teacher fails.
                failures.append("云端 Qwen 失败：{}".format(str(exc)[:300]))
        values = {
            key: shlex.quote(str(value))
            for key, value in {
                "sample_id": sample_id,
                "video": video,
                **paths,
                "repository_root": self.settings.repository_root,
            }.items()
        }
        local_teacher = self.settings.default_teacher_command()
        if local_teacher:
            try:
                teacher_command = render_command(local_teacher, **values)
                with self.teacher_gpu_slot():
                    admission = self.gpus.teacher_availability()
                    if not admission["enabled"]:
                        raise RuntimeError(
                            "Qwen GPU admission rejected: {} card(s) are required, idle GPUs are {}"
                            .format(
                                admission["required_gpu_count"],
                                admission["idle_gpu_ids"],
                            )
                        )
                    self.gpus.set_teacher_active(admission["gpu_ids"])
                    try:
                        self._execute(
                            teacher_command,
                            extra_env=self.gpus.teacher_process_environment(
                                admission["gpu_ids"]
                            ),
                        )
                        self._validate_teacher_artifact(paths["teacher"])
                        return "Qwen 本机教师分析完成"
                    finally:
                        self.gpus.set_teacher_active([])
            except Exception as exc:
                failures.append("本机 Qwen 失败：{}".format(str(exc)[:300]))

        api_status = getattr(
            self.settings,
            "qwen_api_status",
            lambda: {"available": False, "reason": "未配置 Qwen API 备用通道"},
        )()
        if api_status.get("available"):
            try:
                run_qwen_api(
                    getattr(self.settings, "qwen_api")(),
                    sample_id=sample_id,
                    feature=paths["feature"],
                    pose_video=paths["pose_video"],
                    output=paths["teacher"],
                    prediction=paths["prediction"],
                    saved_key=getattr(self.settings, "qwen_api_key", lambda: "")(),
                )
                prefix = "；".join(failures)
                return (
                    (prefix + "；") if prefix else ""
                ) + "已切换至 Qwen API 备用教师分析"
            except Exception as exc:
                failures.append("Qwen API 备用通道失败：{}".format(str(exc)[:300]))
        elif not remote_teacher["enabled"] and not local_teacher:
            failures.append(str(api_status.get("reason") or "未配置 Qwen API 备用通道"))
        raise RuntimeError("；".join(failures) or "没有可用的 Qwen 教师执行通道")

    @staticmethod
    def _validate_teacher_artifact(path: Path) -> None:
        """Reject a transport-successful but unusable teacher result.

        SSH/worker processes can exit successfully yet leave an interrupted or
        non-teacher JSON file.  Treat that as a source failure so the hosted
        fallback is still considered, instead of failing only after the job
        has already committed to the source.
        """
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            result = value["result"]
            if (
                value.get("schema_version") != "teacher_prediction.v1"
                or value.get("status") != "completed"
                or not isinstance(result, dict)
                or not result.get("label")
                or not result.get("distribution")
                or not result.get("evidence")
            ):
                raise ValueError("missing completed teacher prediction fields")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("教师输出文件无效：{}".format(str(exc)[:300])) from exc

    def prediction(self, sample_id: str) -> Optional[dict]:
        path = self.artifacts(sample_id)["prediction"]
        # Existing Campus6 source artifacts are authoritative.  A materialized
        # JSON file may predate the current calibration or rarity metadata.
        if self.baseline is not None and self.baseline.has(sample_id):
            value = self.baseline.prediction(sample_id)
            if value is not None:
                value = dict(value)
                pose_metrics = self.baseline.pose_metrics(sample_id)
                gate = teacher_gate_decision(
                    value,
                    self.settings.teacher_trigger_confidence,
                    pose_quality=pose_metrics["quality"],
                    pose_quality_threshold=self.settings.pose_quality_threshold,
                    instability_score=None,
                    instability_threshold=self.settings.student_instability_threshold,
                )
                gate = self._boolean_teacher_gate(value, gate)
                gate["pose_metrics"] = pose_metrics
                value["teacher_gate"] = gate
                return value
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def hard_samples(self, *, limit: int = 100, offset: int = 0) -> dict:
        """Return samples satisfying Hard(x)=C1 or C2 or C3 or C5."""
        if self.baseline is None:
            return {"total": 0, "items": []}
        from dahua_cup.backend.hard_samples import evaluate_hard_sample

        candidates = []
        teacher_capability = self.capability_state()["qwen_teacher"]
        for sample_id in self.baseline.sample_ids():
            try:
                sample = self.store.get_sample(sample_id)
            except KeyError:
                continue
            # Campus6 initial records carry their source annotation as
            # ``reviewed``.  Do not use workflow status to hide them here:
            # only an explicit label submitted from the hard-sample page
            # moves a sample out of this queue into the durable hard pool.
            if sample.get("incremental_reason") == "human_reviewed_hard_sample":
                continue
            prediction = self.prediction(sample_id)
            if prediction is None:
                continue
            teacher = self.teacher_state(
                sample_id,
                capability=teacher_capability,
                prediction=prediction,
            )
            decision = evaluate_hard_sample(
                prediction,
                teacher,
                confidence_threshold=self.settings.teacher_trigger_confidence,
                conflict_confidence_threshold=(
                    self.settings.teacher_conflict_confidence
                ),
            )
            if not decision["is_hard"]:
                continue
            matched = set(decision["matched_conditions"])
            # This tuple is a discrete queue order, never a hard score.
            priority = (
                "C3" in matched,
                "C2" in matched,
                len(matched),
                "C1" in matched,
                "C5" in matched,
            )
            candidates.append((priority, sample, prediction, teacher, decision))

        candidates.sort(
            key=lambda item: tuple(-int(value) for value in item[0])
            + (item[1]["sample_id"],)
        )
        items = []
        for _, sample, prediction, teacher, decision in candidates[
            offset:offset + limit
        ]:
            sample_id = sample["sample_id"]
            sample.pop("video_path", None)
            sample.pop("hard_score", None)
            sample.pop("quality_score", None)
            items.append({
                **sample,
                "prediction": prediction,
                "teacher": teacher,
                "hard_decision": decision,
                "matched_conditions": decision["matched_conditions"],
                "matched_condition_count": decision[
                    "matched_condition_count"
                ],
                "media": {"pose": f"/api/samples/{sample_id}/media/pose"},
            })
        return {"total": len(candidates), "items": items}

    def pose_video(self, sample_id: str) -> Optional[Path]:
        """Return a skeleton-only video, rendering existing poses if needed."""
        path = self.artifacts(sample_id)["pose_video"]
        if path.is_file() and path.stat().st_size > 0:
            return path
        if self.baseline is not None and self.baseline.has(sample_id):
            return self.baseline.render_pose_video(sample_id, path)
        return None

    def materialize_baseline_inputs(self, sample_id: str) -> Dict[str, Path]:
        """Prepare worker files from existing poses without running RTMPose."""
        paths = self.artifacts(sample_id)
        if self.baseline is None or not self.baseline.has(sample_id):
            return paths
        self.baseline.materialize_feature(sample_id, paths["feature"])
        self.baseline.render_pose_video(sample_id, paths["pose_video"])
        prediction = self.prediction(sample_id)
        if prediction is not None:
            paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
            temporary = paths["prediction"].with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(paths["prediction"])
        return paths

    def student_snapshot(self, sample_id: str) -> dict:
        prediction = self.prediction(sample_id)
        captured_at = _now()
        if prediction is not None:
            prediction_path = self.artifacts(sample_id)["prediction"]
            snapshot = dict(prediction)
            snapshot.update({
                "status": prediction.get("status", "completed"),
                "model": "ProtoGCN",
                "captured_at": captured_at,
                "prediction_path": (
                    str(prediction_path)
                    if prediction_path.is_file()
                    else "existing-campus6-int8-index"
                ),
            })
            snapshot.setdefault(
                "generated_at",
                (
                    datetime.fromtimestamp(
                        prediction_path.stat().st_mtime, timezone.utc
                    ).isoformat()
                    if prediction_path.is_file() else captured_at
                ),
            )
            snapshot["top6"] = list(prediction.get("topk") or [])[:6]
            snapshot.pop("topk", None)
            return snapshot

        sample = self.store.get_sample(sample_id)
        latest = next(
            (
                job for job in sample.get("jobs", [])
                if job.get("action") in {"classify", "full"}
            ),
            None,
        )
        if latest is None:
            return {"status": "not_run", "captured_at": captured_at, "top6": []}
        status = latest.get("status") or "unknown"
        return {
            "status": status,
            "captured_at": captured_at,
            "top6": [],
            "job_id": latest.get("job_id"),
            "device": latest.get("student_device", ""),
            "error": latest.get("message", "") if status == "failed" else "",
        }

    def teacher_state(
        self,
        sample_id: str,
        *,
        capability: Optional[dict] = None,
        prediction: Optional[dict] = None,
    ) -> dict:
        capability = capability or self.capability_state()["qwen_teacher"]
        sample = self.store.get_sample(sample_id)
        prediction = prediction or self.prediction(sample_id) or {}
        gate = prediction.get("teacher_gate") or {}
        decision = evaluate_hard_sample(
            prediction,
            None,
            confidence_threshold=self.settings.teacher_trigger_confidence,
            conflict_confidence_threshold=(
                self.settings.teacher_conflict_confidence
            ),
        )
        if decision["confidence_gate"]["eligible"] is False:
            return {
                "status": "skipped",
                "model": None,
                "reason": decision["summary"],
                "result": None,
                "gate": gate,
            }
        latest = next(
            (
                job for job in sample.get("jobs", [])
                if job.get("action") == "teacher"
                or (
                    job.get("action") == "full"
                    and gate.get("teacher_called")
                )
            ),
            None,
        )
        if latest is not None and latest.get("status") in {"queued", "running", "failed"}:
            status = latest.get("status") or "unknown"
            if (
                status == "failed"
                and latest.get("message")
                == "视频必须位于服务器候选数据或 Web 上传目录内"
            ):
                return {
                    "status": "not_run",
                    "model": "Qwen3-VL-8B-Instruct",
                    "reason": "教师骨架输入校验已更新，等待重新分析",
                    "result": None,
                }
            return {
                "status": status,
                "model": "Qwen3-VL-8B-Instruct",
                "reason": latest.get("message", ""),
                "result": None,
                "job_id": latest.get("job_id"),
            }

        path = self.artifacts(sample_id)["teacher"]
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                return {
                    "status": "failed",
                    "model": None,
                    "reason": f"教师结果无法读取：{exc}",
                    "result": None,
                }
            saved_prompt_version = str(
                (value.get("provenance") or {}).get("prompt_version") or ""
            )
            if saved_prompt_version != PROMPT_VERSION:
                return {
                    "status": "not_run" if capability["enabled"] else "not_enabled",
                    "model": "Qwen3-VL-8B-Instruct" if capability["enabled"] else None,
                    "reason": "教师提示词已更新，旧结果不再使用，等待重新分析",
                    "result": None,
                    "gate": gate,
                }
            value.setdefault("status", "completed")
            value.setdefault("model", None)
            value.setdefault("result", None)
            value["prediction_path"] = str(path)
            return value

        if gate and not gate.get("teacher_called"):
            confidence = gate.get("student_confidence")
            threshold = gate.get("threshold")
            if gate.get("route") == "human_review_unusable_pose":
                reason = (
                    "骨架产物不可用（{}），无法运行 ProtoGCN/Qwen，"
                    "已进入人工复核"
                ).format(
                    (gate.get("pose_metrics") or {}).get(
                        "status", "unknown"
                    )
                )
                status = "blocked_quality"
            elif gate.get("triggered"):
                if capability["enabled"]:
                    reason = "难例已进入 Qwen 队列，等待服务器 GPU 准入"
                    status = "not_run"
                else:
                    reason = "学生结果不确定，但 Qwen 当前未启用"
                    status = "not_enabled"
            else:
                try:
                    confidence_value = float(confidence)
                    threshold_value = float(threshold)
                except (TypeError, ValueError):
                    reason = "当前样本缺少完整的教师路由统计；仍可手工强制运行 Qwen"
                else:
                    reason = (
                        "Campus6 学生 Top-1 置信度 {:.1%} 达到阈值 {:.1%}，"
                        "自动跳过 Qwen；仍可手工强制运行"
                    ).format(confidence_value, threshold_value)
                status = "skipped"
            return {
                "status": status,
                "model": (
                    "Qwen3-VL-8B-Instruct"
                    if capability["enabled"] else None
                ),
                "reason": reason,
                "result": None,
                "gate": gate,
            }

        return {
            "status": "not_run" if capability["enabled"] else "not_enabled",
            "model": "Qwen3-VL-8B-Instruct" if capability["enabled"] else None,
            "reason": capability["reason"],
            "result": None,
        }

    def teacher_snapshot(self, sample_id: str) -> dict:
        snapshot = self.teacher_state(sample_id)
        snapshot["captured_at"] = _now()
        return snapshot

    def artifact_status(self, sample_id: str) -> Dict[str, bool]:
        result = {}
        for key, path in self.artifacts(sample_id).items():
            if key == "work_dir":
                continue
            exists = path.is_file() and path.stat().st_size > 0
            result[key] = exists
        if self.baseline is not None and self.baseline.has(sample_id):
            result["feature"] = True
            result["pose_video"] = True
            result["prediction"] = self.baseline.prediction(sample_id) is not None
        return result
