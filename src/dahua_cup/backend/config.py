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
from dahua_cup.backend.remote import load_qwen_remote, save_qwen_remote


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

    @property
    def gpu_settings_path(self) -> Path:
        return self.runtime_root / "settings" / "gpu.json"

    @property
    def macro_parameters_path(self) -> Path:
        return self.runtime_root / "settings" / "macro_parameters.json"

    @property
    def qwen_remote_path(self) -> Path:
        return self.runtime_root / "settings" / "qwen_remote.json"

    def qwen_remote(self) -> dict:
        return load_qwen_remote(self.qwen_remote_path)

    def update_qwen_remote(self, value: dict) -> dict:
        return save_qwen_remote(self.qwen_remote_path, value)

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
        if self.pose_command:
            return self.pose_command
        python_bin = os.environ.get("DAHUA_RTMPOSE_PYTHON", "").strip()
        if not python_bin or not Path(python_bin).is_file():
            return ""
        return (f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_pose_worker "
                "--video {video} --feature {feature} --device cpu --max-frames 100 "
                f"--joint-score-threshold {self.joint_score_threshold}")

    def default_locator_command(self) -> str:
        return ""

    def resolve_student_checkpoint(self) -> Optional[Path]:
        """Return the accepted M1KD QAT INT8 + Logits KD student."""
        configured = os.environ.get("DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT", "").strip()
        server_default = Path(
            "/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/"
            "m1kd_best_full/M1KD.runtime.int8.pt"
        )
        candidate = (
            Path(configured).expanduser()
            if configured
            else server_default if server_default.is_file()
            else self.repository_root / "models" / "student" / "M1KD.int8.pt"
        )
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None

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
        config = os.environ.get("DAHUA_CAMPUS6_DEPLOY_CONFIG", str(self.repository_root / "third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"))
        labels = CONFIG_ROOT / "campus" / "campus6_labels.txt"
        return (f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_student_worker "
                "--sample-id {sample_id} --feature {feature} --output {prediction} "
                f"--config {shlex.quote(config)} --checkpoint {shlex.quote(str(checkpoint))} --checkpoint-format quantized "
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
