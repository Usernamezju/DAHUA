from dahua_cup.backend.config import Settings


def test_defaults_restore_qat_student_and_lora_incremental_training(tmp_path, monkeypatch):
    student = tmp_path / "models" / "student" / "M1KD.int8.pt"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"qat")
    training = tmp_path / "campus6_gap_fp32.pth"
    training.write_bytes(b"fp32")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("DAHUA_STUDENT_PYTHON", str(python))
    monkeypatch.setenv("DAHUA_CAMPUS6_TRAINING_CHECKPOINT", str(training))
    settings = Settings(
        repository_root=tmp_path,
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        source_root=tmp_path / "runtime" / "videos",
        manifest_path=tmp_path / "runtime" / "manifest.csv",
        database_path=tmp_path / "runtime" / "review.sqlite3",
        artifact_root=tmp_path / "runtime" / "artifacts",
        frontend_root=tmp_path / "frontend",
        pose_command="",
        student_command="",
        ffmpeg=None,
    )

    assert settings.resolve_student_checkpoint() == student.resolve()
    assert settings.deployment_checkpoint_format == "quantized"
    command = settings.default_incremental_train_command()
    assert "--distill" in command
    assert "--lora" in command
    assert "--deep-compression" not in command
