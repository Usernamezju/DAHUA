"""Configuration for the single supported Campus6 deployment path."""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from dahua_cup.paths import CONFIG_ROOT, REPOSITORY_ROOT


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


def _routing_configuration(path: Path, *, required: bool) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"teacher routing config not found: {path}")
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != "teacher_routing.v1":
        raise ValueError("unsupported teacher routing config schema")
    return value


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
    teacher_routing_config: Optional[Path] = None
    joint_score_threshold: float = 0.20
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
        data_root = _path_env("DAHUA_DATA_ROOT", server_data_root if server_data_root.is_dir() else repository_root)
        runtime_root = _path_env("DAHUA_VIS_RUNTIME_ROOT", data_root / "runtime" / "visualization")
        source_root = _path_env("DAHUA_VIS_SOURCE_ROOT", runtime_root / "videos")
        manifest_path = _path_env("DAHUA_VIS_MANIFEST", runtime_root / "manual_label_manifest.csv")
        routing_env = os.environ.get("DAHUA_TEACHER_ROUTING_CONFIG", "").strip()
        routing_path = Path(routing_env).expanduser().resolve() if routing_env else CONFIG_ROOT / "campus" / "teacher_routing.yaml"
        routing = _routing_configuration(routing_path, required=bool(routing_env))
        routing_values = dict(routing.get("routing") or {})
        priorities = dict(routing.get("review_priority") or {})
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
            teacher_trigger_confidence=_unit_interval_env("DAHUA_TEACHER_TRIGGER_CONFIDENCE", float(routing_values.get("confidence_threshold", 0.70))),
            teacher_trigger_margin=_unit_interval_env("DAHUA_TEACHER_TRIGGER_MARGIN", float(routing_values.get("margin_threshold", 0.15))),
            pose_quality_threshold=_unit_interval_env("DAHUA_POSE_QUALITY_THRESHOLD", float(routing_values.get("pose_quality_threshold", 0.70))),
            student_instability_threshold=_unit_interval_env("DAHUA_STUDENT_INSTABILITY_THRESHOLD", float(routing_values.get("instability_threshold", 0.60))),
            teacher_conflict_confidence=_unit_interval_env("DAHUA_TEACHER_CONFLICT_CONFIDENCE", float(routing_values.get("teacher_conflict_confidence", 0.70))),
            priority_teacher_conflict=_nonnegative_int_env("DAHUA_PRIORITY_TEACHER_CONFLICT", int(priorities.get("student_teacher_conflict", 1_000_000))),
            priority_pose_quality=_nonnegative_int_env("DAHUA_PRIORITY_POSE_QUALITY", int(priorities.get("pose_quality_failure", 900_000))),
            priority_student_instability=_nonnegative_int_env("DAHUA_PRIORITY_STUDENT_INSTABILITY", int(priorities.get("student_instability", 800_000))),
            priority_teacher_review=_nonnegative_int_env("DAHUA_PRIORITY_TEACHER_REVIEW", int(priorities.get("teacher_requested_review", 750_000))),
            priority_teacher_unavailable=_nonnegative_int_env("DAHUA_PRIORITY_TEACHER_UNAVAILABLE", int(priorities.get("uncertain_teacher_unavailable", 700_000))),
            max_workers=max(1, _nonnegative_int_env("DAHUA_VIS_MAX_WORKERS", 2)),
        )

    def ensure_directories(self) -> None:
        root = self.data_root.resolve()
        managed = (self.runtime_root, self.database_path, self.artifact_root)
        outside = [str(path) for path in managed if not path_is_within(path, root)]
        if outside:
            raise ValueError("runtime paths must stay under DAHUA_DATA_ROOT: " + ", ".join(outside))
        if not (path_is_within(self.manifest_path, self.source_root) or path_is_within(self.manifest_path, self.runtime_root)):
            raise ValueError("DAHUA_VIS_MANIFEST must stay under the source or runtime root")
        for path in (self.runtime_root, self.upload_root, self.database_path.parent,
                     self.artifact_root / "previews", self.artifact_root / "features",
                     self.artifact_root / "pose_videos", self.artifact_root / "predictions",
                     self.artifact_root / "teachers", self.artifact_root / "work",
                     self.runtime_root / "exports", self.runtime_root / "settings"):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def upload_root(self) -> Path:
        return self.runtime_root / "videos"

    @property
    def gpu_settings_path(self) -> Path:
        return self.runtime_root / "settings" / "gpu.json"

    @property
    def campus6_backend(self) -> bool:
        return True

    def video_path_is_allowed(self, path: Path) -> bool:
        return path_is_within(path, self.source_root) or path_is_within(path, self.upload_root)

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
        configured = os.environ.get("DAHUA_CAMPUS6_CHECKPOINT", "").strip()
        candidate = Path(configured).expanduser() if configured else self.repository_root / "models" / "student" / "campus6_protogcn_gap_fp32_epoch40.pth"
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None

    def default_student_command(self) -> str:
        if self.student_command:
            return self.student_command
        checkpoint = self.resolve_student_checkpoint()
        python_bin = os.environ.get("DAHUA_STUDENT_PYTHON", "").strip()
        if not checkpoint or not python_bin or not Path(python_bin).is_file():
            return ""
        config = os.environ.get("DAHUA_CAMPUS6_CONFIG", str(self.repository_root / "third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"))
        labels = CONFIG_ROOT / "campus" / "campus6_labels.txt"
        return (f"{shlex.quote(python_bin)} -m dahua_cup.pipeline.rtmpose17_student_worker "
                "--sample-id {sample_id} --feature {feature} --output {prediction} "
                f"--config {shlex.quote(config)} --checkpoint {shlex.quote(str(checkpoint))} "
                f"--label-map {shlex.quote(str(labels))} --device {{device}}")

    def default_teacher_command(self) -> str:
        """Qwen is an externally configured service; no snapshot is bundled."""
        return self.teacher_command
