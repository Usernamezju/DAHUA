from pathlib import Path

from dahua_cup.backend.config import Settings


def test_explicit_data_root_is_the_default_runtime_root(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("DAHUA_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("DAHUA_DATA_ROOT", str(runtime))
    for name in (
        "DAHUA_VIS_RUNTIME_ROOT",
        "DAHUA_VIS_SOURCE_ROOT",
        "DAHUA_VIS_DATABASE",
        "DAHUA_VIS_ARTIFACT_ROOT",
        "DAHUA_VIS_MANIFEST",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.runtime_root == runtime.resolve()
    assert settings.source_root == (runtime / "videos").resolve()
    assert settings.database_path == (runtime / "review/review.sqlite3").resolve()
    assert settings.artifact_root == (runtime / "artifacts").resolve()
