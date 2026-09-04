"""Configuration for the single supported Campus6 deployment path."""

from __future__ import annotations

import os
import shlex
import shutil
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from dahua_cup.paths import CONFIG_ROOT, REPOSITORY_ROOT

from dahua_cup.backend.macro_parameters import load_macro_parameter_file
from dahua_cup.backend.pose_backend import (
    load_pose_backend,
    pose_backend_status,
    save_pose_backend,
)
from dahua_cup.backend.remote import load_qwen_remote, save_qwen_remote
from dahua_cup.backend.qwen_api import (
    api_status as qwen_api_status,
    clear_qwen_api_secret,
    load_qwen_api,
    load_qwen_api_secret,
    save_qwen_api,
    save_qwen_api_secret,
)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _unit_interval_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float_env(name: str, default: float) -> float:
    configured = os.environ.get(name, "").strip()
    if not configured:
        return default
    try:
        value = float(configured)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _boolean_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 0/1, true/false, yes/no")


def _routing_configuration(path: Path, *, required: bool) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"teacher routing config not found: {path}")
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != "teacher_routing.v1":
        raise ValueError("unsupported teacher routing config schema")
    return value


@dataclass
class Settings:
    repository_root: Path
    data_root: Path
    runtime_root: Path
    source_root: Path
    manifest_path: Path
    database_path: Path
    artifact_root: Path
    frontend_root: Path
    pose_command: str
    student_command: str
    ffmpeg: Optional[str]
    teacher_command: str = ""
    teacher_routing_config: Optional[Path] = None
    joint_score_threshold: float = 0.20
    teacher_trigger_confidence: float = 0.30
    pose_quality_threshold: float = 0.70
    student_instability_threshold: float = 0.60
    teacher_conflict_confidence: float = 0.70
    review_temperature: float = 5.0
    incremental_batch_size: int = 50
    max_workers: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        repository_root = _path_env("DAHUA_CODE_ROOT", REPOSITORY_ROOT)
        server_data_root = Path("/workspace/data/xzz_data/DAHUA")
        configured_data_root = os.environ.get("DAHUA_DATA_ROOT", "").strip()
        use_server_default = not configured_data_root and server_data_root.is_dir()
        data_root = _path_env(
            "DAHUA_DATA_ROOT",
            server_data_root if use_server_default else repository_root / "runtime",
        )
        runtime_root = _path_env(
            "DAHUA_VIS_RUNTIME_ROOT",
            data_root / "runtime" / "campus6_product"
            if use_server_default else data_root,
        )
        source_root = _path_env("DAHUA_VIS_SOURCE_ROOT", runtime_root / "videos")
        manifest_path = _path_env("DAHUA_VIS_MANIFEST", runtime_root / "campus6_manifest.csv")
        routing_env = os.environ.get("DAHUA_TEACHER_ROUTING_CONFIG", "").strip()
        routing_path = Path(routing_env).expanduser().resolve() if routing_env else CONFIG_ROOT / "campus" / "teacher_routing.yaml"
        routing = _routing_configuration(routing_path, required=bool(routing_env))
        routing_values = dict(routing.get("routing") or {})
        saved_macro = load_macro_parameter_file(
            runtime_root / "settings" / "macro_parameters.json"
        )
        return cls(
            repository_root=repository_root, data_root=data_root, runtime_root=runtime_root,
            source_root=source_root, manifest_path=manifest_path,
            database_path=_path_env("DAHUA_VIS_DATABASE", runtime_root / "review" / "review.sqlite3"),
            artifact_root=_path_env("DAHUA_VIS_ARTIFACT_ROOT", runtime_root / "artifacts"),
            frontend_root=CONFIG_ROOT.parent / "visualization",
            pose_command=os.environ.get("DAHUA_VIS_POSE_COMMAND", "").strip(),
            student_command=os.environ.get("DAHUA_VIS_STUDENT_COMMAND", "").strip(),
            teacher_command=os.environ.get("DAHUA_VIS_TEACHER_COMMAND", "").strip(),
            ffmpeg=shutil.which(os.environ.get("DAHUA_FFMPEG", "ffmpeg")),
            teacher_routing_config=routing_path if routing_path.is_file() else None,
            joint_score_threshold=_unit_interval_env("DAHUA_RTMPOSE_JOINT_SCORE_THRESHOLD", 0.20),
            teacher_trigger_confidence=_unit_interval_env("DAHUA_TEACHER_TRIGGER_CONFIDENCE", float(saved_macro.get("teacher_trigger_confidence", float(routing_values.get("confidence_threshold", 0.30))))),
            pose_quality_threshold=_unit_interval_env("DAHUA_POSE_QUALITY_THRESHOLD", float(routing_values.get("pose_quality_threshold", 0.70))),
            student_instability_threshold=_unit_interval_env("DAHUA_STUDENT_INSTABILITY_THRESHOLD", float(routing_values.get("instability_threshold", 0.60))),
            teacher_conflict_confidence=_unit_interval_env("DAHUA_TEACHER_CONFLICT_CONFIDENCE", float(saved_macro.get("teacher_conflict_confidence", float(routing_values.get("teacher_conflict_confidence", 0.70))))),
            review_temperature=_positive_float_env("DAHUA_CAMPUS6_REVIEW_TEMPERATURE", _positive_float_env("DAHUA_CAMPUS6_PROBABILITY_TEMPERATURE", 5.0)),
            incremental_batch_size=max(1, _nonnegative_int_env("DAHUA_INCREMENTAL_BATCH_SIZE", int(saved_macro.get("incremental_batch_size", 50)))),
            max_workers=max(1, _nonnegative_int_env("DAHUA_VIS_MAX_WORKERS", 2)),
        )

    def ensure_directories(self) -> None:
        root = self.data_root.resolve()
        managed = (self.runtime_root, self.database_path, self.artifact_root)
        outside = [str(path) for path in managed if not path_is_within(path, root)]
        if outside:
            raise ValueError("runtime paths must stay under DAHUA_DATA_ROOT: " + ", ".join(outside))
        if not (path_is_within(self.manifest_path, self.source_root) or path_is_within(self.manifest_path, self.runtime_root)):
            raise ValueError("DAHUA_VIS_MANIFEST must stay under the Campus6 source or runtime root")
        for path in (self.runtime_root, self.database_path.parent,
                     self.artifact_root / "features",
                     self.artifact_root / "pose_videos", self.artifact_root / "predictions",
                     self.artifact_root / "teachers", self.artifact_root / "work",
                     self.runtime_root / "exports", self.runtime_root / "settings",
                     self.runtime_root / "videos"):
            path.mkdir(parents=True, exist_ok=True)
        # Dataset state is intentionally separate from disposable runtime
        # artifacts.  ``campus_all`` is initialized lazily from the shipped
        # baseline by the incremental manager, never regenerated by the Web.
        for path in (
            self.dataset_root / "campus_increment",
            self.dataset_root / "campus_all",
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_root(self) -> Path:
        """Root containing campus6_baseline, campus_increment and campus_all."""
        configured = os.environ.get("DAHUA_DATASET_ROOT", "").strip()
        root = Path(configured).expanduser() if configured else self.repository_root / "dataset"
        return root.resolve()

    @property
    def incremental_seed(self) -> int:
        return _nonnegative_int_env("DAHUA_INCREMENTAL_SEED", 20260827)

    def default_incremental_train_command(self) -> str:
        """Return the local ProtoGCN fine-tuning command for one frozen batch.

        A deployment may override this with ``DAHUA_INCREMENTAL_TRAIN_COMMAND``
        and use the placeholders ``{ann_file}``, ``{work_dir}``, and
        ``{init_checkpoint}``.  The default deliberately uses the local GCN
        environment, not the Qwen environment or any remote model weights.
        """
        configured = os.environ.get("DAHUA_INCREMENTAL_TRAIN_COMMAND", "").strip()
        if configured:
            return configured
        python_bin = os.environ.get("DAHUA_STUDENT_PYTHON", "").strip()
        checkpoint = self.training_baseline_checkpoint()
        if not python_bin or not Path(python_bin).is_file() or checkpoint is None:
            return ""
        return (
            f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.train_campus6 "
            "--ann-file {ann_file} --work-dir {work_dir} "
            f"--init-checkpoint {shlex.quote(str(checkpoint))} --epochs 10 --distill --lora"
        )

    def default_incremental_evaluate_command(self) -> str:
        """Return the built-in candidate/baseline evaluation command."""
        python_bin = os.environ.get("DAHUA_STUDENT_PYTHON", "").strip()
        if not python_bin or not Path(python_bin).is_file():
            return ""
        return (
            f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.evaluate_incremental "
            "--config {config} --candidate-checkpoint {candidate_checkpoint} "
            "--baseline-checkpoint {baseline_checkpoint} --annotation {evaluation_annotation} "
            "--old-annotation {old_annotation} --new-annotation {new_annotation} "
            "--output {metrics_file} --lora"
        )

    @property
    def incremental_evaluate_command(self) -> str:
        return os.environ.get(
            "DAHUA_INCREMENTAL_EVALUATE_COMMAND",
            self.default_incremental_evaluate_command(),
        ).strip()

    @property
    def incremental_model_registry_path(self) -> Path:
        return self.runtime_root / "settings" / "model_registry.jsonl"

    @property
    def incremental_production_pointer_path(self) -> Path:
        return self.runtime_root / "settings" / "production_pointer.json"

    def incremental_release_gates(self) -> dict[str, float | int]:
        """Read release thresholds while keeping safe, conservative defaults."""
        def number(name: str, default: float) -> float:
            configured = os.environ.get(name, "").strip()
            if not configured:
                return default
            try:
                value = float(configured)
            except ValueError as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            return value

        return {
            "minimum_global_macro_f1_delta": number("DAHUA_INCREMENTAL_MIN_GLOBAL_F1_DELTA", 0.0),
            "minimum_new_macro_f1_delta": number("DAHUA_INCREMENTAL_MIN_NEW_F1_DELTA", 0.0),
            "maximum_old_macro_f1_drop": number("DAHUA_INCREMENTAL_MAX_OLD_F1_DROP", 0.02),
            "minimum_dangerous_recall_delta": number("DAHUA_INCREMENTAL_MIN_DANGEROUS_RECALL_DELTA", 0.0),
            "maximum_ece_increase": number("DAHUA_INCREMENTAL_MAX_ECE_INCREASE", 0.02),
            "maximum_edge_size_bytes": int(number("DAHUA_INCREMENTAL_MAX_EDGE_SIZE_BYTES", 50 * 1024 * 1024)),
        }

    def incremental_remote(self) -> dict:
        """Reuse the trusted Qwen SSH endpoint for server-side GCN training."""
        remote = dict(self.qwen_remote())
        enabled = _boolean_env(
            "DAHUA_INCREMENTAL_REMOTE_ENABLED", bool(remote.get("enabled"))
        )
        remote["enabled"] = enabled and bool(remote.get("enabled"))
        return remote

    @property
    def incremental_remote_python(self) -> str:
        return os.environ.get(
            "DAHUA_INCREMENTAL_REMOTE_PYTHON",
            "/root/miniconda3/envs/skel_gcn38/bin/python",
        ).strip()

    @property
    def incremental_remote_init_checkpoint(self) -> str:
        return os.environ.get(
            "DAHUA_INCREMENTAL_REMOTE_INIT_CHECKPOINT",
            "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
            "deployment_benchmark_20260829/M0.deployment.fp32.pth",
        ).strip()

    @property
    def incremental_remote_gpu_id(self) -> str:
        return os.environ.get("DAHUA_INCREMENTAL_REMOTE_GPU_ID", "auto").strip()

    @property
    def local_inference_device(self) -> str:
        """Select local worker hardware: ``auto``, ``cpu``, ``gpu`` or server.

        ``auto`` first leases an available configured CUDA device and falls
        back to CPU only when no device is visible.  The local launcher resolves
        this before Web startup using both worker environments, so it does not
        select a GPU that one of the two Python environments cannot use.
        ``DAHUA_LOCAL_CPU_INFERENCE=1`` remains a backwards-compatible CPU
        override for older launch scripts.
        """
        if _boolean_env("DAHUA_LOCAL_CPU_INFERENCE", False):
            return "cpu"
        value = os.environ.get("DAHUA_LOCAL_INFERENCE_DEVICE", "server").strip().lower()
        if value not in {"server", "auto", "cpu", "gpu"}:
            raise ValueError(
                "DAHUA_LOCAL_INFERENCE_DEVICE must be server, auto, cpu or gpu"
            )
        return value

    @property
    def local_cpu_inference(self) -> bool:
        """Compatibility alias for integrations that only need a CPU flag."""
        return self.local_inference_device == "cpu"

    @property
    def gpu_settings_path(self) -> Path:
        return self.runtime_root / "settings" / "gpu.json"

    @property
    def macro_parameters_path(self) -> Path:
        return self.runtime_root / "settings" / "macro_parameters.json"

    @property
    def qwen_remote_path(self) -> Path:
        return self.runtime_root / "settings" / "qwen_remote.json"

    @property
    def qwen_api_path(self) -> Path:
        """Persist non-secret hosted-Qwen fallback settings only."""
        return self.runtime_root / "settings" / "qwen_api.json"

    @property
    def qwen_api_secret_path(self) -> Path:
        """Owner-restricted server-local key entered through the Web page."""
        return self.runtime_root / "settings" / "qwen_api.secret"

    @property
    def pose_backend_path(self) -> Path:
        return self.runtime_root / "settings" / "pose_backend.json"

    @property
    def pose_int8_model_root(self) -> Path:
        configured = os.environ.get("DAHUA_POSE_INT8_MODEL_ROOT", "").strip()
        root = Path(configured).expanduser() if configured else self.repository_root / "models" / "pose" / "int8"
        return root.resolve()

    @property
    def pose_fp16_model_root(self) -> Path:
        configured = os.environ.get("DAHUA_POSE_FP16_MODEL_ROOT", "").strip()
        root = Path(configured).expanduser() if configured else self.repository_root / "models" / "pose" / "fp16"
        return root.resolve()

    def pose_backend(self) -> str:
        return load_pose_backend(self.pose_backend_path)

    def pose_backend_status(self) -> dict:
        return pose_backend_status(
            self.pose_backend_path,
            self.pose_int8_model_root,
            fp32_available=self._fp32_pose_available(),
            fp16_root=self.pose_fp16_model_root,
        )

    def update_pose_backend(self, backend_id: str) -> dict:
        return save_pose_backend(
            self.pose_backend_path,
            backend_id,
            self.pose_int8_model_root,
            fp32_available=self._fp32_pose_available(),
            fp16_root=self.pose_fp16_model_root,
        )

    def _fp32_pose_available(self) -> bool:
        if self.pose_command:
            return True
        python_bin = os.environ.get("DAHUA_RTMPOSE_PYTHON", "").strip()
        return bool(python_bin and Path(python_bin).is_file())

    def qwen_remote(self) -> dict:
        return load_qwen_remote(self.qwen_remote_path)

    def update_qwen_remote(self, value: dict) -> dict:
        return save_qwen_remote(self.qwen_remote_path, value)

    def qwen_api(self) -> dict:
        return load_qwen_api(self.qwen_api_path)

    def qwen_api_status(self) -> dict:
        return qwen_api_status(self.qwen_api(), self.qwen_api_key())

    def qwen_api_key(self) -> str:
        return load_qwen_api_secret(self.qwen_api_secret_path)

    def update_qwen_api(self, value: dict) -> dict:
        payload = dict(value)
        api_key = str(payload.pop("api_key", "") or "").strip()
        clear_key = bool(payload.pop("clear_api_key", False))
        cleaned = save_qwen_api(self.qwen_api_path, payload)
        if clear_key:
            clear_qwen_api_secret(self.qwen_api_secret_path)
        elif api_key:
            save_qwen_api_secret(self.qwen_api_secret_path, api_key)
        return qwen_api_status(cleaned, self.qwen_api_key())

    @property
    def campus6_backend(self) -> bool:
        return True

    def video_path_is_allowed(self, path: Path) -> bool:
        # Registered baseline samples point at derived skeleton videos in the
        # private runtime artifact store, never at source RGB media.
        return (
            path_is_within(path, self.source_root)
            or path_is_within(path, self.artifact_root / "pose_videos")
        )

    @property
    def preview_codec(self) -> str:
        return os.environ.get("DAHUA_VIS_PREVIEW_CODEC", "libx264").strip() or "libx264"

    @property
    def preview_preset(self) -> str:
        return os.environ.get("DAHUA_VIS_PREVIEW_PRESET", "veryfast").strip()

    @property
    def preview_bitrate(self) -> str:
        return os.environ.get("DAHUA_VIS_PREVIEW_BITRATE", "2M").strip() or "2M"

    def default_pose_command(self) -> str:
        backend = self.pose_backend()
        if self.pose_command and backend == "mmpose_fp32":
            return self.pose_command
        python_bin = os.environ.get("DAHUA_RTMPOSE_PYTHON", "").strip()
        if not python_bin or not Path(python_bin).is_file():
            return ""
        if backend in {"tensorrt_fp16", "tensorrt_int8"}:
            status = self.pose_backend_status()
            if not status["selected_available"]:
                return ""
            if backend == "tensorrt_fp16":
                model_root = self.pose_fp16_model_root
                detector_dir = "rtmdet_nano_person"
            else:
                model_root = self.pose_int8_model_root
                detector_dir = "rtmdet_s"
            return (
                f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_pose_worker "
                "--video {video} --feature {feature} --device cuda:0 --max-frames 100 "
                "--backend mmdeploy "
                f"--detector-model-dir {shlex.quote(str(model_root / detector_dir))} "
                f"--pose-model-dir {shlex.quote(str(model_root / 'rtmpose_s'))} "
                f"--joint-score-threshold {self.joint_score_threshold}"
            )
        pose_weights = self.repository_root / "models" / "pose" / "rtmpose-s_coco17.pth"
        detector_weights = self.repository_root / "models" / "pose" / "rtmdet-s_coco80.pth"
        offline_weights = ""
        if pose_weights.is_file() and detector_weights.is_file():
            offline_weights = (
                f" --pose2d-weights {shlex.quote(str(pose_weights))}"
                f" --det-weights {shlex.quote(str(detector_weights))}"
            )
        return (f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_pose_worker "
                "--video {video} --feature {feature} --device cpu --max-frames 100 --backend mmpose "
                "--extractor-id rtm-s-coco17.v1 "
                f"--joint-score-threshold {self.joint_score_threshold}{offline_weights}")

    def default_locator_command(self) -> str:
        return ""

    def resolve_student_checkpoint(self) -> Optional[Path]:
        """Return the selected Campus6 deployment student checkpoint.

        The default is the accepted M1KD QAT-plus-logit-distillation INT8
        checkpoint.  Deep Compression remains available only when explicitly
        selected through the deployment environment.
        """
        configured = os.environ.get("DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT", "").strip()
        server_default = Path(
            "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
            "m1kd_best_full/M1KD.runtime.int8.pt"
        )
        bundled_default = self.repository_root / "models" / "student" / "M1KD.int8.pt"
        candidate = Path(configured).expanduser() if configured else None
        if candidate is None:
            pointer_path = self.incremental_production_pointer_path
            registry_path = self.incremental_model_registry_path
            if pointer_path.is_file() and registry_path.is_file():
                try:
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                    current_id = str(pointer.get("current_model_id") or "")
                    records = [
                        json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    matches = [row for row in records if str(row.get("model_id")) == current_id and row.get("status") == "production"]
                    if matches and matches[-1].get("checkpoint_path"):
                        candidate = Path(str(matches[-1]["checkpoint_path"])).expanduser()
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    candidate = None
        if candidate is None:
            candidate = server_default if server_default.is_file() else bundled_default
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None

    def resolve_student_residual_checkpoint(self) -> Optional[Path]:
        """Return the non-Huffman state required by the compressed student."""
        checkpoint = self.resolve_student_checkpoint()
        if checkpoint is None or checkpoint.suffix != ".bin":
            return None
        configured = os.environ.get("DAHUA_CAMPUS6_DEPLOYMENT_RESIDUAL_CHECKPOINT", "").strip()
        if checkpoint.name.endswith(".huffman.bin"):
            stem = checkpoint.name.removesuffix(".huffman.bin")
            # The inference sidecar deliberately omits GAP/CSC tensors used
            # only by training.  Keep the earlier full sidecar as a fallback
            # for deployments that have not yet copied the slimmer artifact.
            preferred_residual = checkpoint.with_name(
                stem + ".inference.residual.pth"
            )
            legacy_residual = checkpoint.with_name(stem + ".residual.pth")
            default_residual = (
                preferred_residual
                if preferred_residual.is_file()
                else legacy_residual
            )
        else:
            default_residual = checkpoint.with_suffix(checkpoint.suffix + ".residual.pth")
        candidate = (
            Path(configured).expanduser()
            if configured
            else default_residual
        ).resolve()
        return candidate if candidate.is_file() else None

    @property
    def deployment_checkpoint_format(self) -> str:
        """Checkpoint loader format for the selected production student."""
        value = os.environ.get(
            "DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT_FORMAT", ""
        ).strip().lower()
        if not value:
            checkpoint = self.resolve_student_checkpoint()
            return "huffman" if checkpoint is not None and checkpoint.suffix == ".bin" else "quantized"
        if value not in {"auto", "fp32", "quantized", "huffman"}:
            raise ValueError(
                "DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT_FORMAT must be auto, fp32, quantized or huffman"
            )
        return value

    def training_baseline_checkpoint(self) -> Optional[Path]:
        """Return the GAP FP32 checkpoint retained for training/regression."""
        configured = os.environ.get("DAHUA_CAMPUS6_TRAINING_CHECKPOINT", "").strip()
        server_default = Path(
            "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/"
            "campus6_rtmpose26_k400_2d_gap_full_manual_v2/"
            "best_top1_acc_epoch_40.pth"
        )
        candidate = (
            Path(configured).expanduser()
            if configured
            else server_default if server_default.is_file()
            else self.repository_root / "models" / "student" / "campus6_protogcn_gap_fp32_epoch40.pth"
        )
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None

    def baseline_annotations(self) -> Optional[Path]:
        """Existing Campus6 COCO-17 skeletons and official manual labels."""
        configured = os.environ.get(
            "DAHUA_CAMPUS6_BASELINE_ANNOTATIONS", ""
        ).strip()
        server_default = Path(
            "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
            "m1kd_best_full/annotations_with_all.pkl"
        )
        candidate = (
            Path(configured).expanduser()
            if configured else server_default
        ).resolve()
        return candidate if candidate.is_file() else None

    def baseline_predictions(self) -> Optional[Path]:
        """Per-sample probabilities produced once by the accepted INT8 artifact."""
        configured = os.environ.get(
            "DAHUA_CAMPUS6_BASELINE_PREDICTIONS", ""
        ).strip()
        server_default = Path(
            "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
            "m1kd_best_full/M1KD.eval_all.pkl"
        )
        candidate = (
            Path(configured).expanduser()
            if configured else server_default
        ).resolve()
        return candidate if candidate.is_file() else None

    def student_probability_temperature(self) -> float:
        """Return the total temperature used for review-facing probabilities.

        The accepted cache retains its validation-fitted temperature as
        provenance.  Review uses a deliberately softer total temperature of
        5.0 by default, while the legacy environment variables remain
        compatible overrides.  The value is loaded at construction time from
        the persisted macro-parameter file and can be updated live through
        ``PUT /api/macro-parameters``.
        """
        return self.review_temperature

    def default_student_command(self) -> str:
        if self.student_command:
            return self.student_command
        checkpoint = self.resolve_student_checkpoint()
        python_bin = os.environ.get("DAHUA_STUDENT_PYTHON", "").strip()
        if not checkpoint or not python_bin or not Path(python_bin).is_file():
            return ""
        residual = self.resolve_student_residual_checkpoint()
        if self.deployment_checkpoint_format == "huffman" and residual is None:
            return ""
        config = os.environ.get("DAHUA_CAMPUS6_DEPLOY_CONFIG", str(self.repository_root / "third_party/ProtoGCN/configs/campus6/rtm_s_coco17_k400_2d_gap_full.py"))
        labels = CONFIG_ROOT / "campus" / "campus6_labels.txt"
        return (f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_student_worker "
                "--sample-id {sample_id} --feature {feature} --output {prediction} "
                f"--config {shlex.quote(config)} --checkpoint {shlex.quote(str(checkpoint))} "
                f"--checkpoint-format {shlex.quote(self.deployment_checkpoint_format)} "
                f"{('--residual-checkpoint ' + shlex.quote(str(residual)) + ' ') if residual else ''}"
                f"--label-map {shlex.quote(str(labels))} --device {{device}} "
                "--temperature {student_temperature}")

    def default_teacher_command(self) -> str:
        """Build an on-server Qwen command without downloading model weights.

        An explicit command still wins, which permits a remote inference
        service.  Otherwise a deployment running on the competition server
        discovers the shared Qwen3-VL-8B snapshot and invokes it lazily.
        """
        if self.teacher_command:
            return self.teacher_command
        default_model = Path("/workspace/data/public_data/Qwen3-VL-8B-Instruct")
        model_dir = Path(
            os.environ.get("DAHUA_QWEN_MODEL_DIR", str(default_model))
        ).expanduser()
        default_python = Path("/workspace/code/envs/llm_env/bin/python")
        teacher_python = Path(
            os.environ.get("DAHUA_TEACHER_PYTHON", str(default_python))
        ).expanduser()
        if not model_dir.is_dir() or not teacher_python.is_file():
            return ""
        labels = CONFIG_ROOT / "campus" / "campus6_labels.txt"
        dtype = os.environ.get("DAHUA_QWEN_DTYPE", "float16").strip()
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("DAHUA_QWEN_DTYPE must be float16, bfloat16 or float32")
        return (
            f"{shlex.quote(str(teacher_python))} -m dahua_cup.pipeline.qwen_teacher_worker "
            "--sample-id {sample_id} --feature {feature} --pose-video {pose_video} "
            "--student-json {prediction} --output {teacher} "
            f"--model-dir {shlex.quote(str(model_dir))} "
            f"--label-map {shlex.quote(str(labels))} --dtype {shlex.quote(dtype)} "
            "--device-map auto --attn-implementation sdpa --max-frames 8 --max-new-tokens 512"
        )
