"""Environment-backed configuration for the visualization server."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def path_is_within(path: Path, root: Path) -> bool:
    """Return whether path resolves under root on Python 3.8 and newer."""
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


def _positive_int_env(name: str, default: int) -> int:
    value = _nonnegative_int_env(name, default)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
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


def _repair_split_checkpoint(command: str, checkpoint: Path) -> str:
    """Repair an unquoted ``--checkpoint /dir/ file.pth`` deployment typo."""
    if not checkpoint.is_file():
        return command
    try:
        arguments = shlex.split(command)
        option_index = arguments.index("--checkpoint")
    except (ValueError, IndexError):
        return command
    if option_index + 2 >= len(arguments):
        return command
    split_path = Path(
        arguments[option_index + 1] + arguments[option_index + 2]
    ).expanduser()
    if split_path.resolve() != checkpoint.resolve():
        return command
    arguments[option_index + 1] = str(checkpoint)
    del arguments[option_index + 2]
    return shlex.join(arguments)


@dataclass(frozen=True)
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
    locator_command: str = ""
    evolution_enabled: bool = False
    evolution_config: Optional[Path] = None
    evolution_root: Optional[Path] = None
    model_registry_path: Optional[Path] = None
    production_pointer_path: Optional[Path] = None
    ntu120_label_map: Optional[Path] = None
    pose_detection_confidence: float = 0.50
    pose_presence_confidence: float = 0.50
    pose_tracking_confidence: float = 0.50
    joint_score_threshold: float = 0.20
    teacher_routing_config: Optional[Path] = None
    teacher_trigger_confidence: float = 0.70
    teacher_trigger_margin: float = 0.15
    pose_quality_threshold: float = 0.70
    student_instability_threshold: float = 0.60
    teacher_conflict_confidence: float = 0.70
    priority_teacher_conflict: int = 1_000_000
    priority_pose_quality: int = 900_000
    priority_student_instability: int = 800_000
    priority_teacher_review: int = 750_000
    priority_teacher_unavailable: int = 700_000
    max_workers: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        repository_root = _path_env("DAHUA_CODE_ROOT", REPOSITORY_ROOT)
        server_data_root = Path("/workspace/data/xzz_data/DAHUA")
        default_data_root = (
            server_data_root if server_data_root.is_dir() else repository_root
        )
        data_root = _path_env("DAHUA_DATA_ROOT", default_data_root)
        runtime_root = _path_env(
            "DAHUA_VIS_RUNTIME_ROOT", data_root / "runtime" / "visualization"
        )
        dataset_candidates = data_root / "datasets" / "vedio"
        legacy_candidates = data_root / "manual_label_candidates"
        if (dataset_candidates / "manual_label_manifest.csv").is_file():
            default_source = dataset_candidates
        elif legacy_candidates.is_dir():
            default_source = legacy_candidates
        else:
            default_source = runtime_root / "videos"
        source_root = _path_env("DAHUA_VIS_SOURCE_ROOT", default_source)
        source_manifest = source_root / "manual_label_manifest.csv"
        default_manifest = (
            source_manifest if source_manifest.is_file()
            else runtime_root / "manual_label_manifest.csv"
        )
        manifest_path = _path_env(
            "DAHUA_VIS_MANIFEST", default_manifest
        )
        routing_env = os.environ.get("DAHUA_TEACHER_ROUTING_CONFIG", "").strip()
        routing_path = Path(routing_env).expanduser().resolve() if routing_env else (
            repository_root
            / "dahua_cup"
            / "configs"
            / "campus"
            / "teacher_routing.yaml"
        )
        routing_config = _routing_configuration(
            routing_path, required=bool(routing_env)
        )
        routing = dict(routing_config.get("routing") or {})
        priorities = dict(routing_config.get("review_priority") or {})
        evolution_env = os.environ.get(
            "DAHUA_AUTO_EVOLUTION_CONFIG", ""
        ).strip()
        evolution_path = (
            Path(evolution_env).expanduser().resolve()
            if evolution_env
            else repository_root
            / "dahua_cup"
            / "configs"
            / "campus"
            / "auto_evolution_ntu120.yaml"
        )
        evolution_flag = os.environ.get("DAHUA_AUTO_EVOLUTION", "1").strip()
        if evolution_flag not in {"0", "1"}:
            raise ValueError("DAHUA_AUTO_EVOLUTION must be 0 or 1")
        if evolution_env and not evolution_path.is_file():
            raise FileNotFoundError(
                f"auto-evolution config not found: {evolution_path}"
            )
        evolution_root = runtime_root / "evolution"
        return cls(
            repository_root=repository_root,
            data_root=data_root,
            runtime_root=runtime_root,
            source_root=source_root,
            manifest_path=manifest_path,
            database_path=_path_env(
                "DAHUA_VIS_DATABASE", runtime_root / "review" / "review.sqlite3"
            ),
            artifact_root=_path_env(
                "DAHUA_VIS_ARTIFACT_ROOT", runtime_root / "artifacts"
            ),
            frontend_root=repository_root / "dahua_cup" / "visualization",
            pose_command=os.environ.get("DAHUA_VIS_POSE_COMMAND", "").strip(),
            student_command=os.environ.get("DAHUA_VIS_STUDENT_COMMAND", "").strip(),
            ffmpeg=shutil.which(os.environ.get("DAHUA_FFMPEG", "ffmpeg")),
            teacher_command=os.environ.get("DAHUA_VIS_TEACHER_COMMAND", "").strip(),
            locator_command=os.environ.get(
                "DAHUA_VIS_LOCATOR_COMMAND", ""
            ).strip(),
            evolution_enabled=(
                evolution_flag == "1" and evolution_path.is_file()
            ),
            evolution_config=(
                evolution_path if evolution_path.is_file() else None
            ),
            evolution_root=_path_env(
                "DAHUA_EVOLUTION_ROOT", evolution_root
            ),
            model_registry_path=_path_env(
                "DAHUA_MODEL_REGISTRY",
                evolution_root / "models" / "registry.jsonl",
            ),
            production_pointer_path=_path_env(
                "DAHUA_PRODUCTION_POINTER",
                evolution_root / "models" / "production.json",
            ),
            ntu120_label_map=_path_env(
                "DAHUA_NTU120_LABEL_MAP",
                repository_root
                / "gcn_models"
                / "GAP"
                / "text"
                / "ntu120_label_map.txt",
            ),
            pose_detection_confidence=_unit_interval_env(
                "DAHUA_MEDIAPIPE_DETECTION_CONFIDENCE", 0.50
            ),
            pose_presence_confidence=_unit_interval_env(
                "DAHUA_MEDIAPIPE_PRESENCE_CONFIDENCE", 0.50
            ),
            pose_tracking_confidence=_unit_interval_env(
                "DAHUA_MEDIAPIPE_TRACKING_CONFIDENCE", 0.50
            ),
            joint_score_threshold=_unit_interval_env(
                "DAHUA_MEDIAPIPE_JOINT_SCORE_THRESHOLD", 0.20
            ),
            teacher_routing_config=(
                routing_path if routing_path.is_file() else None
            ),
            teacher_trigger_confidence=_unit_interval_env(
                "DAHUA_TEACHER_TRIGGER_CONFIDENCE",
                float(routing.get("confidence_threshold", 0.70)),
            ),
            teacher_trigger_margin=_unit_interval_env(
                "DAHUA_TEACHER_TRIGGER_MARGIN",
                float(routing.get("margin_threshold", 0.15)),
            ),
            pose_quality_threshold=_unit_interval_env(
                "DAHUA_POSE_QUALITY_THRESHOLD",
                float(routing.get("pose_quality_threshold", 0.70)),
            ),
            student_instability_threshold=_unit_interval_env(
                "DAHUA_STUDENT_INSTABILITY_THRESHOLD",
                float(routing.get("instability_threshold", 0.60)),
            ),
            teacher_conflict_confidence=_unit_interval_env(
                "DAHUA_TEACHER_CONFLICT_CONFIDENCE",
                float(routing.get("teacher_conflict_confidence", 0.70)),
            ),
            priority_teacher_conflict=_nonnegative_int_env(
                "DAHUA_PRIORITY_TEACHER_CONFLICT",
                int(priorities.get("student_teacher_conflict", 1_000_000)),
            ),
            priority_pose_quality=_nonnegative_int_env(
                "DAHUA_PRIORITY_POSE_QUALITY",
                int(priorities.get("pose_quality_failure", 900_000)),
            ),
            priority_student_instability=_nonnegative_int_env(
                "DAHUA_PRIORITY_STUDENT_INSTABILITY",
                int(priorities.get("student_instability", 800_000)),
            ),
            priority_teacher_review=_nonnegative_int_env(
                "DAHUA_PRIORITY_TEACHER_REVIEW",
                int(priorities.get("teacher_requested_review", 750_000)),
            ),
            priority_teacher_unavailable=_nonnegative_int_env(
                "DAHUA_PRIORITY_TEACHER_UNAVAILABLE",
                int(priorities.get("uncertain_teacher_unavailable", 700_000)),
            ),
            max_workers=max(1, int(os.environ.get("DAHUA_VIS_MAX_WORKERS", "2"))),
        )

    def ensure_directories(self) -> None:
        root = self.data_root.resolve()
        managed_paths = (
            self.runtime_root,
            self.database_path,
            self.artifact_root,
            self.evolution_root or self.runtime_root,
            self.model_registry_path or self.runtime_root,
            self.production_pointer_path or self.runtime_root,
        )
        outside = [str(path) for path in managed_paths if not path_is_within(path, root)]
        if outside:
            raise ValueError(
                "visualization runtime paths must stay under DAHUA_DATA_ROOT: "
                + ", ".join(outside)
            )
        if not (
            path_is_within(self.manifest_path, self.source_root)
            or path_is_within(self.manifest_path, self.runtime_root)
        ):
            raise ValueError(
                "DAHUA_VIS_MANIFEST must stay under the source or runtime root"
            )
        for path in (
            self.runtime_root,
            self.upload_root,
            self.database_path.parent,
            self.artifact_root / "previews",
            self.artifact_root / "localized_videos",
            self.artifact_root / "localization",
            self.artifact_root / "features",
            self.artifact_root / "pose_videos",
            self.artifact_root / "predictions",
            self.artifact_root / "teachers",
            self.runtime_root / "exports",
            self.runtime_root / "settings",
            self.evolution_root or self.runtime_root / "evolution",
            (
                self.model_registry_path.parent
                if self.model_registry_path
                else self.runtime_root / "evolution" / "models"
            ),
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def upload_root(self) -> Path:
        return self.runtime_root / "videos"

    @property
    def gpu_settings_path(self) -> Path:
        return self.runtime_root / "settings" / "gpu.json"

    def video_path_is_allowed(self, path: Path) -> bool:
        return path_is_within(path, self.source_root) or path_is_within(
            path, self.upload_root
        )

    @property
    def preview_codec(self) -> str:
        return (
            os.environ.get("DAHUA_VIS_PREVIEW_CODEC", "libx264").strip()
            or "libx264"
        )

    @property
    def preview_preset(self) -> str:
        return os.environ.get("DAHUA_VIS_PREVIEW_PRESET", "veryfast").strip()

    @property
    def preview_bitrate(self) -> str:
        return os.environ.get("DAHUA_VIS_PREVIEW_BITRATE", "2M").strip() or "2M"

    def default_pose_command(self) -> str:
        if self.campus6_backend:
            pose_python = os.environ.get("DAHUA_RTMPOSE_PYTHON", "/root/miniconda3/envs/rtmpose26/bin/python").strip()
            if not Path(pose_python).is_file():
                return ""
            return (
                f"{shlex.quote(pose_python)} -m dahua_cup.pipeline.rtmpose17_pose_worker "
                "--video {video} --feature {feature} --device cpu --max-frames 100 "
                f"--joint-score-threshold {self.joint_score_threshold}"
            )
        model = os.environ.get("DAHUA_MEDIAPIPE_MODEL", "").strip()
        if not model:
            candidate = (
                self.data_root / "models" / "pose_models" / "mediapipe"
                / "pose_landmarker_heavy.task"
            )
            if candidate.is_file():
                model = str(candidate)
        if self.pose_command:
            return self.pose_command
        pose_python = os.environ.get("DAHUA_POSE_PYTHON", "").strip()
        if not pose_python:
            llm_python = (
                self.repository_root.parent / "envs" / "llm_env" / "bin" / "python"
            )
            if llm_python.is_file():
                pose_python = str(llm_python)
            elif importlib.util.find_spec("mediapipe") is not None:
                pose_python = sys.executable
        if not model or not pose_python:
            return ""
        return (
            f"{shlex.quote(pose_python)} -m dahua_cup.pipeline.mediapipe_pose_worker "
            f"--video {{video}} --feature {{feature}} --model {shlex.quote(model)} "
            "--num-poses 2 --delegate cpu "
            "--min-pose-detection-confidence "
            f"{self.pose_detection_confidence} "
            "--min-pose-presence-confidence "
            f"{self.pose_presence_confidence} "
            "--min-tracking-confidence "
            f"{self.pose_tracking_confidence} "
            "--joint-score-threshold "
            f"{self.joint_score_threshold}"
        )

    def default_locator_command(self) -> str:
        """Return an optional NanoDet group-crop command.

        The locator is enabled only when both official NanoDet files are
        configured. Existing deployments therefore keep their original
        MediaPipe-only behavior until NanoDet is intentionally installed.
        """
        enabled = os.environ.get("DAHUA_NANODET_ENABLED", "0").strip()
        if enabled not in {"0", "1"}:
            raise ValueError("DAHUA_NANODET_ENABLED must be 0 or 1")
        if enabled == "0":
            return ""
        if self.locator_command:
            return self.locator_command
        config_value = os.environ.get("DAHUA_NANODET_CONFIG", "").strip()
        checkpoint_value = os.environ.get(
            "DAHUA_NANODET_CHECKPOINT", ""
        ).strip()
        if not config_value or not checkpoint_value:
            return ""
        config = Path(config_value).expanduser().resolve()
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not config.is_file() or not checkpoint.is_file():
            return ""
        python_bin = os.environ.get("DAHUA_NANODET_PYTHON", "").strip()
        if not python_bin:
            if importlib.util.find_spec("nanodet") is None:
                return ""
            python_bin = sys.executable
        confidence = _unit_interval_env("DAHUA_NANODET_CONFIDENCE", 0.35)
        minimum_area = _unit_interval_env(
            "DAHUA_NANODET_MINIMUM_AREA_FRACTION", 0.0001
        )
        minimum_crop = _unit_interval_env(
            "DAHUA_NANODET_MINIMUM_CROP_FRACTION", 0.20
        )
        context_factor = float(
            os.environ.get("DAHUA_NANODET_CONTEXT_FACTOR", "1.20")
        )
        if context_factor < 1.0:
            raise ValueError(
                "DAHUA_NANODET_CONTEXT_FACTOR must be at least one"
            )
        output_size = int(
            os.environ.get("DAHUA_NANODET_OUTPUT_SIZE", "640")
        )
        if output_size < 32:
            raise ValueError("DAHUA_NANODET_OUTPUT_SIZE must be at least 32")
        crowd_policy = os.environ.get(
            "DAHUA_NANODET_CROWD_POLICY", "full-frame"
        ).strip()
        if crowd_policy not in {"top2", "full-frame"}:
            raise ValueError(
                "DAHUA_NANODET_CROWD_POLICY must be top2 or full-frame"
            )
        device = os.environ.get("DAHUA_NANODET_DEVICE", "cpu").strip()
        return (
            f"{shlex.quote(python_bin)} -m "
            "dahua_cup.pipeline.nanodet_group_crop_worker "
            "--video {video} --output {localized_video} "
            "--metadata {localization} "
            f"--config {shlex.quote(str(config))} "
            f"--checkpoint {shlex.quote(str(checkpoint))} "
            f"--device {shlex.quote(device)} "
            f"--confidence {confidence} "
            f"--minimum-area-fraction {minimum_area} "
            f"--context-factor {context_factor} "
            f"--minimum-crop-fraction {minimum_crop} "
            f"--output-size {output_size} "
            f"--crowd-policy {crowd_policy} "
            f"--ffmpeg {shlex.quote(str(self.ffmpeg or 'ffmpeg'))} "
            f"--codec {shlex.quote(self.preview_codec)} "
            f"--preset {shlex.quote(self.preview_preset)} "
            f"--bitrate {shlex.quote(self.preview_bitrate)}"
        )

    def default_student_command(self) -> str:
        if self.campus6_backend:
            checkpoint = self.resolve_student_checkpoint()
            student_python = os.environ.get("DAHUA_STUDENT_PYTHON", "/root/miniconda3/envs/skel_gcn38/bin/python").strip()
            if not checkpoint or not Path(student_python).is_file():
                return ""
            config = os.environ.get(
                "DAHUA_CAMPUS6_CONFIG",
                str(self.repository_root / "gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"),
            )
            labels = self.repository_root / "dahua_cup/configs/campus/campus6_labels.txt"
            return (
                f"{shlex.quote(student_python)} -m dahua_cup.pipeline.rtmpose17_student_worker "
                "--sample-id {sample_id} --feature {feature} --output {prediction} "
                f"--config {shlex.quote(config)} --checkpoint {shlex.quote(str(checkpoint))} "
                f"--label-map {shlex.quote(str(labels))} --device {{device}}"
            )
        default_candidate = (
            self.data_root / "models" / "protogcn_pretrained"
            / "ntu120_xsub_bone_b1" / "best_top1_acc_epoch_150.pth"
        )
        checkpoint_path = self.resolve_student_checkpoint()
        checkpoint = str(checkpoint_path) if checkpoint_path else ""
        if self.student_command:
            return _repair_split_checkpoint(
                self.student_command, default_candidate
            )
        if (
            not checkpoint
            or importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("mmcv") is None
        ):
            return ""
        return (
            f"{sys.executable} -m dahua_cup.pipeline.protogcn_student_worker "
            f"--sample-id {{sample_id}} --feature {{feature}} "
            f"--output {{prediction}} "
            f"--checkpoint {shlex.quote(checkpoint)} --device {{device}}"
        )

    def resolve_student_checkpoint(self) -> Optional[Path]:
        """Resolve a gate-promoted checkpoint, falling back to immutable base."""
        if self.campus6_backend:
            configured = os.environ.get("DAHUA_CAMPUS6_CHECKPOINT", "").strip()
            candidate = Path(configured).expanduser() if configured else (
                self.data_root / "experiments/ProtoGCN/campus6_rtmpose26_k400_2d_gap_full_manual_v2/best_top1_acc_epoch_40.pth"
            )
            candidate = candidate.resolve()
            return candidate if candidate.is_file() else None
        if (
            self.production_pointer_path
            and self.model_registry_path
            and self.production_pointer_path.is_file()
            and self.model_registry_path.is_file()
        ):
            try:
                pointer = json.loads(
                    self.production_pointer_path.read_text(encoding="utf-8")
                )
                model_id = pointer.get("current_model_id")
                records = [
                    json.loads(line)
                    for line in self.model_registry_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                history = [
                    row for row in records if row.get("model_id") == model_id
                ]
                if history:
                    promoted = Path(
                        str(history[-1].get("checkpoint_path", ""))
                    ).expanduser().resolve()
                    if promoted.is_file():
                        return promoted
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        configured = os.environ.get(
            "DAHUA_PROTOGCN_PRETRAINED", ""
        ).strip()
        if configured:
            candidate = Path(configured).expanduser().resolve()
            return candidate if candidate.is_file() else None
        candidate = (
            self.data_root
            / "models"
            / "protogcn_pretrained"
            / "ntu120_xsub_bone_b1"
            / "best_top1_acc_epoch_150.pth"
        )
        return candidate.resolve() if candidate.is_file() else None

    def default_teacher_command(self) -> str:
        if self.teacher_command:
            return self.teacher_command
        public_qwen = Path(
            "/workspace/data/public_data/Qwen3-VL-8B-Instruct"
        )
        default_qwen = (
            public_qwen
            if public_qwen.is_dir()
            else self.data_root / "models" / "Qwen3-VL-8B-Instruct"
        )
        model_dir = _path_env(
            "DAHUA_QWEN_MODEL_DIR",
            default_qwen,
        )
        dtype = os.environ.get("DAHUA_QWEN_DTYPE", "float16").strip()
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "DAHUA_QWEN_DTYPE must be float16, bfloat16 or float32"
            )
        device_map = os.environ.get(
            "DAHUA_QWEN_DEVICE_MAP", "auto"
        ).strip()
        if not device_map:
            raise ValueError("DAHUA_QWEN_DEVICE_MAP must not be empty")
        attention = os.environ.get(
            "DAHUA_QWEN_ATTN_IMPLEMENTATION", "sdpa"
        ).strip()
        if attention not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError(
                "DAHUA_QWEN_ATTN_IMPLEMENTATION must be sdpa, eager or "
                "flash_attention_2"
            )
        max_new_tokens = _positive_int_env(
            "DAHUA_QWEN_MAX_NEW_TOKENS", 512
        )
        max_frames = _positive_int_env("DAHUA_QWEN_MAX_FRAMES", 8)
        if max_frames < 2:
            raise ValueError("DAHUA_QWEN_MAX_FRAMES must be at least two")
        retries = _nonnegative_int_env("DAHUA_QWEN_RETRIES", 2)
        teacher_python = os.environ.get("DAHUA_TEACHER_PYTHON", "").strip()
        if not teacher_python:
            candidate = self.repository_root.parent / "envs" / "llm_env" / "bin" / "python"
            if candidate.is_file():
                teacher_python = str(candidate)
            elif importlib.util.find_spec("transformers") is not None:
                teacher_python = sys.executable
        if not model_dir.is_dir() or not teacher_python:
            return ""
        label_map = (self.repository_root / "dahua_cup/configs/campus/campus6_labels.txt") if self.campus6_backend else (
            self.repository_root / "gcn_models/GAP/text/ntu120_label_map.txt"
        )
        return (
            f"{shlex.quote(teacher_python)} -m dahua_cup.pipeline.qwen_teacher_worker "
            "--sample-id {sample_id} --feature {feature} "
            "--pose-video {pose_video} --student-json {prediction} "
            f"--output {{teacher}} --model-dir {shlex.quote(str(model_dir))} "
            f"--label-map {shlex.quote(str(label_map))} "
            f"--dtype {shlex.quote(dtype)} "
            f"--device-map {shlex.quote(device_map)} "
            f"--attn-implementation {shlex.quote(attention)} "
            f"--max-new-tokens {max_new_tokens} "
            f"--max-frames {max_frames} --retries {retries}"
        )

    @property
    def campus6_backend(self) -> bool:
        """Whether Web jobs should use the direct COCO-17 Campus6 pipeline."""
        value = os.environ.get("DAHUA_CAMPUS6_BACKEND", "0").strip().lower()
        if value not in {"0", "1", "false", "true", "rtmpose17"}:
            raise ValueError("DAHUA_CAMPUS6_BACKEND must be 0/1 or rtmpose17")
        return value in {"1", "true", "rtmpose17"}
