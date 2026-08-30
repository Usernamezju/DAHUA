import csv
import json
import shlex
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dahua_cup.backend.app import _sample_payload, create_app
from dahua_cup.backend.config import Settings
from dahua_cup.backend.store import ReviewStore
from dahua_cup.pipeline.render_ntu25_pose import load_render_data
from dahua_cup.pipeline.video_encoding import browser_video_args


def make_settings(tmp_path):
    runtime = tmp_path / "runtime" / "visualization"
    frontend = tmp_path / "visualization"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>test</h1>", encoding="utf-8")
    return Settings(
        repository_root=tmp_path,
        data_root=tmp_path,
        runtime_root=runtime,
        source_root=runtime / "videos",
        manifest_path=runtime / "manual_label_manifest.csv",
        database_path=runtime / "review" / "review.sqlite3",
        artifact_root=runtime / "artifacts",
        frontend_root=frontend,
        pose_command="",
        student_command="",
        ffmpeg="/usr/bin/ffmpeg",
        max_workers=1,
    )


def write_manifest(settings, video):
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    with settings.manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "clip_id", "path", "source_dataset", "source_label",
            "suggested_coarse_label", "manual_label", "keep", "notes",
        ))
        writer.writeheader()
        writer.writerow({
            "clip_id": "sample-1", "path": video, "source_dataset": "TEST",
            "source_label": "walking", "suggested_coarse_label": "normal_walk",
            "manual_label": "", "keep": "", "notes": "",
        })


def test_web_review_flow_and_export(tmp_path):
    settings = make_settings(tmp_path)
    settings.source_root.mkdir(parents=True)
    video = settings.source_root / "sample.mp4"
    video.write_bytes(b"fake-video")
    write_manifest(settings, video)
    settings.ensure_directories()
    store = ReviewStore(settings.database_path)
    assert store.import_manifest(settings.manifest_path) == {"inserted": 1, "existing": 0}
    assert store.dashboard()["statuses"] == {"pending": 1}
    assert store.get_sample("sample-1")["suggested_label"] == "normal_walk"

    reviewed = store.submit_review(
        "sample-1", "alice", "normal_walk", "accept_suggestion", "clear"
    )
    assert reviewed["status"] == "reviewed"
    relocated = settings.source_root / "relocated.mp4"
    relocated.write_bytes(b"relocated-video")
    write_manifest(settings, relocated)
    assert store.import_manifest(settings.manifest_path) == {"inserted": 0, "existing": 1}
    refreshed = store.get_sample("sample-1")
    assert refreshed["video_path"] == str(relocated)
    assert refreshed["status"] == "reviewed"
    assert refreshed["manual_label"] == "normal_walk"
    assert store.undo_last_review("sample-1", "alice")["status"] == "pending"

    uploaded = settings.source_root / "uploaded.avi"
    uploaded.write_bytes(b"uploaded-video")
    assert store.add_sample("uploaded-1", uploaded)["status"] == "pending"
    escalated = store.escalate_hard_sample(
        "uploaded-1",
        hard_score=1.0,
        priority=1_000_000,
        reason="high_confidence_student_teacher_conflict",
    )
    assert escalated["hard_score"] == 1.0
    assert escalated["priority"] == 1_000_000
    assert store.recent_events(1)[0]["event_type"] == "hard_sample_escalated"
    output = store.export_manifest(settings.runtime_root / "exports" / "labels.csv")
    assert output.is_file()


def test_review_freezes_student_and_teacher_snapshots(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    video = settings.source_root / "snapshot.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("snapshot-1", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    prediction_path = manager.artifacts("snapshot-1")["prediction"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = {
        "schema_version": "protogcn_prediction.v1",
        "task": "ntu120_xsub",
        "modality": "bone",
        "checkpoint_sha256": "abc123",
        "topk": [
            {"class_index": index, "label": "class-{}".format(index), "score": 0.5 - index * 0.05}
            for index in range(5)
        ],
    }
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    try:
        reviewed = store.submit_review(
            "snapshot-1",
            "alice",
            "conflict_chase",
            "intent_conflict",
            student_snapshot=manager.student_snapshot("snapshot-1"),
            teacher_snapshot=manager.teacher_snapshot("snapshot-1"),
        )
        snapshot = reviewed["reviews"][0]
        assert snapshot["student_snapshot"]["status"] == "completed"
        assert snapshot["student_snapshot"]["top5"] == prediction["topk"]
        assert snapshot["student_snapshot"]["checkpoint_sha256"] == "abc123"
        assert snapshot["teacher_snapshot"]["status"] == "not_enabled"

        prediction["topk"][0]["label"] = "changed-later"
        prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
        unchanged = store.get_sample("snapshot-1")["reviews"][0]
        assert unchanged["student_snapshot"]["top5"][0]["label"] == "class-0"

        output = store.export_manifest(settings.runtime_root / "exports" / "snapshot.csv")
        with output.open("r", encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert row["manual_label"] == "conflict_chase"
        assert row["student_status"] == "completed"
        assert row["student_top1_label"] == "class-0"
        assert row["student_checkpoint_sha256"] == "abc123"
        assert row["teacher_status"] == "not_enabled"
    finally:
        manager.executor.shutdown(wait=False)


def test_teacher_state_explains_high_confidence_gate_skip(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    video = settings.source_root / "gate.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("gate-1", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    prediction_path = manager.artifacts("gate-1")["prediction"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps(
            {
                "topk": [{"label": "walking_towards", "score": 0.82}],
                "teacher_gate": {
                    "threshold": 0.70,
                    "student_confidence": 0.82,
                    "triggered": False,
                    "teacher_available": True,
                    "teacher_called": False,
                },
            }
        ),
        encoding="utf-8",
    )
    stale_teacher = manager.artifacts("gate-1")["teacher"]
    stale_teacher.parent.mkdir(parents=True, exist_ok=True)
    stale_teacher.write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {"label": "stale-result"},
            }
        ),
        encoding="utf-8",
    )
    try:
        state = manager.teacher_state("gate-1")
        assert state["status"] == "skipped"
        assert state["result"] is None
        assert "82.0%" in state["reason"]
        assert state["gate"]["threshold"] == 0.70
    finally:
        manager.executor.shutdown(wait=False)


def test_review_store_migrates_existing_review_table(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE reviews (
              review_id INTEGER PRIMARY KEY AUTOINCREMENT,
              sample_id TEXT NOT NULL,
              reviewer TEXT NOT NULL,
              previous_status TEXT NOT NULL,
              previous_label TEXT,
              final_label TEXT NOT NULL,
              reason_code TEXT NOT NULL,
              note TEXT NOT NULL,
              created_at TEXT NOT NULL,
              reverted_at TEXT
            )
            """
        )

    ReviewStore(database)

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(reviews)")}
    assert "student_snapshot_json" in columns
    assert "teacher_snapshot_json" in columns


def test_web_app_exposes_frontend_and_api_routes(tmp_path):
    app = create_app(make_settings(tmp_path))
    paths = {route.path for route in app.routes}
    assert "/api/dashboard" in paths
    assert "/api/samples/{sample_id}/review" in paths
    assert "/api/samples/{sample_id}/jobs" in paths
    assert "/api/gpus" in paths
    assert "/api/evolution" in paths
    assert "/api/evolution/run" in paths
    assert "/api/datasets/export" in paths
    assert "/{frontend_path:path}" in paths


def test_job_manager_deduplicates_active_sample_jobs(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        pose_command=(
            "python -m dahua_cup.pipeline.mediapipe_pose_worker "
            "--video {video} --feature {feature}"
        ),
    )
    settings.ensure_directories()
    video = settings.source_root / "deduplicated.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("deduplicated", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    scheduled = []
    monkeypatch.setattr(
        manager.executor, "submit", lambda function, job_id: scheduled.append(
            (function, job_id)
        )
    )
    try:
        first = manager.submit("deduplicated", "preview")
        repeated = manager.submit("deduplicated", "preview")

        assert not first["reused"]
        assert repeated["reused"]
        assert repeated["job_id"] == first["job_id"]
        assert len(scheduled) == 1
        with pytest.raises(ValueError, match="已有 preview 任务"):
            manager.submit("deduplicated", "pose")
    finally:
        manager.executor.shutdown(wait=False)


def test_existing_mp4_is_available_as_direct_preview(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    video = settings.source_root / "direct.mp4"
    video.write_bytes(b"mp4")
    store = ReviewStore(settings.database_path)
    sample = store.add_sample("direct-mp4", video)
    app = create_app(settings)
    app.state.settings = settings
    app.state.store = store
    from dahua_cup.backend.jobs import JobManager

    app.state.jobs = JobManager(settings, store)
    try:
        payload = _sample_payload(app, sample)
        assert payload["artifacts"]["preview"] is True
    finally:
        app.state.jobs.executor.shutdown(wait=False)


def test_full_job_routes_empty_pose_to_review_before_student(
    tmp_path, monkeypatch
):
    settings = replace(
        make_settings(tmp_path),
        pose_command=(
            "python -m dahua_cup.pipeline.mediapipe_pose_worker "
            "--video {video} --feature {feature} --delegate {delegate}"
        ),
        student_command=(
            "python -m dahua_cup.pipeline.protogcn_student_worker "
            "--feature {feature} --output {prediction} --device {device}"
        ),
    )
    settings.ensure_directories()
    video = settings.source_root / "no-pose.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("no-pose", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    paths = manager.artifacts("no-pose")
    called_student = False

    def fake_execute(command, _extra_env=None):
        nonlocal called_student
        if any(
            value.endswith("mediapipe_pose_worker") for value in command
        ):
            np.savez_compressed(
                paths["feature"],
                schema_version=np.asarray("mediapipe_ntu25.v1"),
                keypoint=np.zeros((2, 4, 25, 3), dtype=np.float32),
                keypoint_score=np.zeros((2, 4, 25), dtype=np.float32),
                valid_mask=np.zeros((2, 4, 25), dtype=bool),
                fps=np.asarray(25.0, dtype=np.float32),
            )
        elif any(value.endswith("render_ntu25_pose") for value in command):
            paths["pose_video"].write_bytes(b"pose-video")
        elif any(
            value.endswith("protogcn_student_worker") for value in command
        ):
            called_student = True
        return ""

    monkeypatch.setattr(manager, "_execute", fake_execute)
    job = store.create_job(
        "no-pose",
        "full",
        pose_device="cpu",
        student_device="cuda:0",
    )
    try:
        manager._run(job["job_id"])
        completed = store.get_job(job["job_id"])
        prediction = manager.prediction("no-pose")
        sample = store.get_sample("no-pose")
        assert completed["status"] == "completed"
        assert "骨架产物不可用" in completed["message"]
        assert not called_student
        assert prediction["status"] == "blocked_quality"
        assert (
            prediction["teacher_gate"]["route"]
            == "human_review_unusable_pose"
        )
        assert prediction["teacher_gate"]["pose_metrics"]["status"] == "no_pose"
        assert sample["priority"] == settings.priority_pose_quality
        assert manager.student_snapshot("no-pose")["status"] == "blocked_quality"
    finally:
        manager.executor.shutdown(wait=False)


def test_full_job_runs_student_and_teacher_for_low_but_usable_pose(
    tmp_path, monkeypatch
):
    settings = replace(
        make_settings(tmp_path),
        pose_command=(
            "python -m dahua_cup.pipeline.mediapipe_pose_worker "
            "--video {video} --feature {feature} --delegate {delegate}"
        ),
        student_command=(
            "python -m dahua_cup.pipeline.protogcn_student_worker "
            "--feature {feature} --output {prediction} --device {device}"
        ),
        teacher_command=(
            "python -m dahua_cup.pipeline.qwen_teacher_worker "
            "--feature {feature} --pose-video {pose_video} "
            "--output {teacher}"
        ),
    )
    settings.ensure_directories()
    video = settings.source_root / "low-pose.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("low-pose", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    paths = manager.artifacts("low-pose")
    called = {"student": False, "teacher": False}

    def fake_execute(command, _extra_env=None):
        if any(
            value.endswith("mediapipe_pose_worker") for value in command
        ):
            valid = np.zeros((2, 4, 25), dtype=bool)
            valid[0, :, :10] = True
            scores = np.zeros((2, 4, 25), dtype=np.float32)
            scores[valid] = 0.5
            np.savez_compressed(
                paths["feature"],
                schema_version=np.asarray("mediapipe_ntu25.v1"),
                keypoint=np.zeros((2, 4, 25, 3), dtype=np.float32),
                keypoint_score=scores,
                valid_mask=valid,
                fps=np.asarray(25.0, dtype=np.float32),
            )
        elif any(value.endswith("render_ntu25_pose") for value in command):
            paths["pose_video"].write_bytes(b"pose-video")
        elif any(
            value.endswith("protogcn_student_worker") for value in command
        ):
            called["student"] = True
            paths["prediction"].write_text(
                json.dumps(
                    {
                        "sample_id": "low-pose",
                        "feature": str(paths["feature"]),
                        "topk": [
                            {"label": "walking", "score": 0.90},
                            {"label": "running", "score": 0.10},
                        ],
                    }
                ),
                encoding="utf-8",
            )
        elif any(value.endswith("qwen_teacher_worker") for value in command):
            called["teacher"] = True
            paths["teacher"].write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "result": {
                            "label": "walking",
                            "confidence": 0.90,
                            "needs_review": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
        return ""

    monkeypatch.setattr(manager, "_execute", fake_execute)
    job = store.create_job(
        "low-pose",
        "full",
        pose_device="cpu",
        student_device="cuda:0",
    )
    try:
        manager._run(job["job_id"])
        completed = store.get_job(job["job_id"])
        prediction = manager.prediction("low-pose")
        sample = store.get_sample("low-pose")

        assert completed["status"] == "completed"
        assert called == {"student": True, "teacher": True}
        assert prediction["teacher_gate"]["route"] == "call_teacher"
        assert prediction["teacher_gate"]["teacher_called"]
        assert (
            "pose_quality_below_threshold"
            in prediction["teacher_gate"]["reasons"]
        )
        assert sample["priority"] == 0
        assert sample["hard_score"] is None
        assert not any(
            event["event_type"] == "hard_sample_escalated"
            for event in store.recent_events(20)
        )
    finally:
        manager.executor.shutdown(wait=False)


def test_web_runtime_must_stay_under_data_root(tmp_path):
    settings = make_settings(tmp_path)
    outside = tmp_path.parent / "outside-runtime"
    invalid = Settings(
        repository_root=settings.repository_root,
        data_root=settings.data_root,
        runtime_root=outside,
        source_root=outside / "videos",
        manifest_path=outside / "manifest.csv",
        database_path=outside / "review.sqlite3",
        artifact_root=outside / "artifacts",
        frontend_root=settings.frontend_root,
        pose_command="",
        student_command="",
        ffmpeg=settings.ffmpeg,
    )
    with pytest.raises(ValueError, match="DAHUA_DATA_ROOT"):
        invalid.ensure_directories()


def test_web_runtime_defaults_below_configured_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "server-data"
    source_root = data_root / "manual_label_candidates"
    source_root.mkdir(parents=True)
    manifest = source_root / "manual_label_manifest.csv"
    manifest.write_text("clip_id,path\n", encoding="utf-8")
    monkeypatch.setenv("DAHUA_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.setenv("DAHUA_DATA_ROOT", str(data_root))
    for name in (
        "DAHUA_VIS_RUNTIME_ROOT",
        "DAHUA_VIS_SOURCE_ROOT",
        "DAHUA_VIS_MANIFEST",
        "DAHUA_VIS_DATABASE",
        "DAHUA_VIS_ARTIFACT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.data_root == data_root.resolve()
    assert settings.runtime_root == (data_root / "runtime" / "visualization").resolve()
    assert settings.source_root == source_root.resolve()
    assert settings.manifest_path == manifest.resolve()
    assert settings.database_path == (
        data_root / "runtime" / "visualization" / "review" / "review.sqlite3"
    ).resolve()


def test_web_prefers_downloaded_vedio_candidate_manifest(tmp_path, monkeypatch):
    data_root = tmp_path / "server-data"
    source_root = data_root / "datasets" / "vedio"
    source_root.mkdir(parents=True)
    manifest = source_root / "manual_label_manifest.csv"
    manifest.write_text("clip_id,path\n", encoding="utf-8")
    legacy = data_root / "manual_label_candidates"
    legacy.mkdir()
    monkeypatch.setenv("DAHUA_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.setenv("DAHUA_DATA_ROOT", str(data_root))
    for name in (
        "DAHUA_VIS_SOURCE_ROOT",
        "DAHUA_VIS_MANIFEST",
        "DAHUA_VIS_RUNTIME_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.source_root == source_root.resolve()
    assert settings.manifest_path == manifest.resolve()


def test_teacher_routing_yaml_and_environment_override(tmp_path, monkeypatch):
    config = tmp_path / "routing.yaml"
    config.write_text(
        """
schema_version: teacher_routing.v1
routing:
  confidence_threshold: 0.61
  margin_threshold: 0.12
review_priority:
  student_teacher_conflict: 4321
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DAHUA_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.setenv("DAHUA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DAHUA_TEACHER_ROUTING_CONFIG", str(config))
    monkeypatch.setenv("DAHUA_TEACHER_TRIGGER_MARGIN", "0.22")

    settings = Settings.from_env()

    assert settings.teacher_trigger_confidence == pytest.approx(0.61)
    assert settings.teacher_trigger_margin == pytest.approx(0.22)
    assert settings.priority_teacher_conflict == 4321
    assert settings.teacher_routing_config == config.resolve()


def test_pose_command_uses_sibling_llm_environment(tmp_path):
    repository_root = tmp_path / "code" / "DAHUA"
    repository_root.mkdir(parents=True)
    settings = make_settings(repository_root)
    pose_python = tmp_path / "code" / "envs" / "llm_env" / "bin" / "python"
    pose_python.parent.mkdir(parents=True)
    pose_python.write_text("", encoding="utf-8")
    model = (
        settings.data_root / "models" / "pose_models" / "mediapipe"
        / "pose_landmarker_heavy.task"
    )
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    command = settings.default_pose_command()

    assert command.startswith(
        str(pose_python) + " -m dahua_cup.pipeline.mediapipe_pose_worker"
    )
    assert str(model) in command
    assert "--num-poses 2 --delegate cpu" in command
    assert "--min-pose-detection-confidence 0.5" in command
    assert "--joint-score-threshold 0.2" in command


def test_teacher_command_exposes_qwen_runtime_profile(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    model = tmp_path / "models" / "Qwen3-VL-8B-Instruct"
    model.mkdir(parents=True)
    python_bin = tmp_path / "envs" / "llm" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    monkeypatch.setenv("DAHUA_QWEN_MODEL_DIR", str(model))
    monkeypatch.setenv("DAHUA_TEACHER_PYTHON", str(python_bin))
    monkeypatch.setenv("DAHUA_QWEN_DTYPE", "float16")
    monkeypatch.setenv("DAHUA_QWEN_DEVICE_MAP", "auto")
    monkeypatch.setenv("DAHUA_QWEN_ATTN_IMPLEMENTATION", "sdpa")
    monkeypatch.setenv("DAHUA_QWEN_MAX_FRAMES", "8")
    monkeypatch.setenv("DAHUA_QWEN_MAX_NEW_TOKENS", "256")
    monkeypatch.setenv("DAHUA_QWEN_RETRIES", "1")

    command = settings.default_teacher_command()

    assert str(model.resolve()) in command
    assert "--dtype float16" in command
    assert "--device-map auto" in command
    assert "--attn-implementation sdpa" in command
    assert "--max-new-tokens 256" in command
    assert "--max-frames 8 --retries 1" in command


def test_nanodet_locator_command_is_optional_and_explicit(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    config = tmp_path / "nanodet-m.yml"
    checkpoint = tmp_path / "nanodet-m.pth"
    python_bin = tmp_path / "bin" / "python"
    config.write_text("model: {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    python_bin.parent.mkdir()
    python_bin.write_text("", encoding="utf-8")
    monkeypatch.setenv("DAHUA_NANODET_CONFIG", str(config))
    monkeypatch.setenv("DAHUA_NANODET_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DAHUA_NANODET_PYTHON", str(python_bin))
    monkeypatch.setenv("DAHUA_NANODET_DEVICE", "cpu")

    monkeypatch.delenv("DAHUA_NANODET_ENABLED", raising=False)
    assert settings.default_locator_command() == ""

    monkeypatch.setenv("DAHUA_NANODET_ENABLED", "1")
    command = settings.default_locator_command()

    assert "dahua_cup.pipeline.nanodet_group_crop_worker" in command
    assert "--output {localized_video}" in command
    assert "--metadata {localization}" in command
    assert "--crowd-policy full-frame" in command
    assert "smoothing" not in command

    monkeypatch.setenv("DAHUA_NANODET_ENABLED", "0")
    assert settings.default_locator_command() == ""


def test_pose_job_runs_nanodet_before_mediapipe(tmp_path, monkeypatch):
    monkeypatch.setenv("DAHUA_NANODET_ENABLED", "1")
    settings = replace(
        make_settings(tmp_path),
        locator_command=(
            "python -m dahua_cup.pipeline.nanodet_group_crop_worker "
            "--video {video} --output {localized_video} "
            "--metadata {localization}"
        ),
        pose_command=(
            "python -m dahua_cup.pipeline.mediapipe_pose_worker "
            "--video {video} --feature {feature} --delegate {delegate}"
        ),
    )
    settings.ensure_directories()
    video = settings.source_root / "localized-first.mp4"
    video.write_bytes(b"video")
    store = ReviewStore(settings.database_path)
    store.add_sample("localized-first", video)
    from dahua_cup.backend.jobs import JobManager

    manager = JobManager(settings, store)
    paths = manager.artifacts("localized-first")
    commands = []

    def fake_preview(_source, output, _logs):
        output.write_bytes(b"preview")

    def fake_execute(command, _extra_env=None):
        commands.append(command)
        if "dahua_cup.pipeline.nanodet_group_crop_worker" in command:
            paths["localized_video"].write_bytes(b"localized")
            paths["localization"].write_text("{}", encoding="utf-8")
        elif "dahua_cup.pipeline.mediapipe_pose_worker" in command:
            video_index = command.index("--video") + 1
            assert command[video_index] == str(paths["localized_video"])
            np.savez_compressed(
                paths["feature"],
                schema_version=np.asarray("mediapipe_ntu25.v1"),
                keypoint=np.zeros((2, 1, 25, 3), dtype=np.float32),
                keypoint_score=np.zeros((2, 1, 25), dtype=np.float32),
                valid_mask=np.zeros((2, 1, 25), dtype=bool),
                fps=np.asarray(25.0, dtype=np.float32),
            )
        elif "dahua_cup.pipeline.render_ntu25_pose" in command:
            paths["pose_video"].write_bytes(b"pose")
        return ""

    monkeypatch.setattr(manager, "_ensure_preview", fake_preview)
    monkeypatch.setattr(manager, "_execute", fake_execute)
    job = store.create_job(
        "localized-first",
        "pose",
        pose_device="cpu",
        student_device="",
    )
    try:
        manager._run(job["job_id"])
        completed = store.get_job(job["job_id"])
        assert completed["status"] == "completed"
        assert "dahua_cup.pipeline.nanodet_group_crop_worker" in commands[0]
        assert "dahua_cup.pipeline.mediapipe_pose_worker" in commands[1]
    finally:
        manager.executor.shutdown(wait=False)


def test_student_command_repairs_split_default_checkpoint(tmp_path):
    settings = make_settings(tmp_path)
    checkpoint = (
        settings.data_root / "models" / "protogcn_pretrained"
        / "ntu120_xsub_bone_b1" / "best_top1_acc_epoch_150.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    settings = replace(
        settings,
        student_command=(
            "python -m dahua_cup.pipeline.protogcn_student_worker "
            "--feature {feature} --output {prediction} "
            "--checkpoint {}/ {} --device {device}"
        ).format(checkpoint.parent, checkpoint.name, feature="{feature}",
                 prediction="{prediction}", device="{device}"),
    )

    arguments = shlex.split(settings.default_student_command())

    checkpoint_index = arguments.index("--checkpoint")
    assert arguments[checkpoint_index + 1] == str(checkpoint)
    assert checkpoint.name not in arguments[checkpoint_index + 2:]


def test_web_preview_codec_defaults_to_libx264(tmp_path, monkeypatch):
    monkeypatch.delenv("DAHUA_VIS_PREVIEW_CODEC", raising=False)
    settings = make_settings(tmp_path)
    assert settings.preview_codec == "libx264"
    assert settings.preview_preset == "veryfast"
    assert settings.preview_bitrate == "2M"
    monkeypatch.setenv("DAHUA_VIS_PREVIEW_CODEC", "h264_nvenc")
    monkeypatch.setenv("DAHUA_VIS_PREVIEW_PRESET", "p4")
    assert settings.preview_codec == "h264_nvenc"
    assert settings.preview_preset == "p4"


def test_browser_video_args_support_conda_openh264():
    openh264 = browser_video_args("libopenh264", "veryfast", bitrate="2M")
    assert openh264[:4] == ["-c:v", "libopenh264", "-b:v", "2M"]
    assert "-preset" not in openh264
    assert "-crf" not in openh264

    x264 = browser_video_args("libx264", "veryfast", quality=24)
    assert x264[:6] == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]


def test_ntu25_renderer_prefers_image_space_points(tmp_path):
    feature = tmp_path / "pose.npz"
    keypoint = np.ones((2, 3, 25, 3), dtype=np.float32)
    image_keypoint = np.zeros((2, 3, 25, 2), dtype=np.float32)
    image_keypoint[..., 0] = 320
    image_keypoint[..., 1] = 180
    np.savez_compressed(
        feature,
        keypoint=keypoint,
        image_keypoint=image_keypoint,
        valid_mask=np.ones((2, 3, 25), dtype=bool),
        fps=np.asarray(25.0),
        width=np.asarray(640),
        height=np.asarray(360),
    )
    points, valid, fps = load_render_data(feature, 960, 540)
    assert points.shape == (2, 3, 25, 2)
    assert np.allclose(points[..., 0], 480)
    assert np.allclose(points[..., 1], 270)
    assert valid.all()
    assert fps == 25.0
