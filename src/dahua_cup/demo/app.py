"""FastAPI entry point for the standalone demo web."""

from __future__ import annotations

import mimetypes
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from dahua_cup.paths import CONFIG_ROOT, REPOSITORY_ROOT
from dahua_cup.demo import pipeline
from dahua_cup.demo.data import DemoDataset


def _path_is_within(path: Path, root: Path) -> bool:
    """Python 3.8-compatible path containment check."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass
class DemoSettings:
    """Demo runtime paths, all overridable through environment variables."""

    dataset_root: Path
    pose_python: Optional[str]
    device: str
    temperature: float
    student_config: Path
    student_checkpoint: Path
    label_map: Path
    frontend_root: Path

    @classmethod
    def from_env(cls) -> "DemoSettings":
        configured_root = os.environ.get("DAHUA_DEMO_DATASET", "").strip()
        dataset_root = Path(configured_root).expanduser() if configured_root \
            else REPOSITORY_ROOT / "dataset"
        return cls(
            dataset_root=dataset_root.resolve(),
            pose_python=os.environ.get("DAHUA_DEMO_POSE_PYTHON", "").strip() or None,
            device=os.environ.get("DAHUA_DEMO_DEVICE", "cuda:0"),
            temperature=float(os.environ.get("DAHUA_DEMO_TEMPERATURE", "1.0")),
            student_config=Path(os.environ.get(
                "DAHUA_DEMO_STUDENT_CONFIG",
                str(REPOSITORY_ROOT / pipeline.PROTOGCN_DEPLOY_CONFIG))),
            student_checkpoint=Path(os.environ.get(
                "DAHUA_DEMO_STUDENT_CHECKPOINT",
                str(REPOSITORY_ROOT / pipeline.STUDENT_CHECKPOINT))),
            label_map=Path(os.environ.get(
                "DAHUA_DEMO_LABEL_MAP",
                str(CONFIG_ROOT / "campus/campus6_labels.txt"))),
            frontend_root=Path(__file__).resolve().parent / "static",
        )


class ImportManager:
    """In-memory upload jobs with a polling-friendly status endpoint."""

    def __init__(self, settings: DemoSettings, dataset: DemoDataset):
        self.settings = settings
        self.dataset = dataset
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, source: Path, original_name: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "sample_id": f"imported_{job_id}",
            "status": "queued",
            "message": "",
            "source": source,
            "original_name": original_name,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=pipeline.run_import_job,
            args=(job, self.dataset, self.settings),
            daemon=True,
        )
        thread.start()
        return job_id

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "message": job.get("message", ""),
            "sample": job.get("sample"),
        }


def create_app(settings: Optional[DemoSettings] = None) -> FastAPI:
    settings = settings or DemoSettings.from_env()
    dataset = DemoDataset(settings.dataset_root)
    jobs = ImportManager(settings, dataset)
    app = FastAPI(title="Campus6 Behavior Recognition Demo")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "dataset": str(settings.dataset_root)}

    @app.get("/api/samples")
    def samples():
        return {"groups": dataset.groups()}

    @app.get("/api/samples/{sample_id}")
    def sample(sample_id: str):
        try:
            row = dataset.get(sample_id)
            prediction = dataset.prediction(sample_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown sample") from None
        return {
            **{key: row.get(key) for key in
               ("id", "label", "person", "scenario", "split", "original_name",
                "imported_at", "group", "source")},
            "media": {
                "rgb": f"/api/media/{sample_id}/rgb",
                "pose": f"/api/media/{sample_id}/pose",
            },
            "prediction": prediction,
        }

    @app.get("/api/media/{sample_id}/{kind}")
    def media(sample_id: str, kind: str):
        if kind not in ("rgb", "pose"):
            raise HTTPException(status_code=404, detail="unknown media kind")
        try:
            path = dataset.media_path(sample_id, kind)
        except KeyError:
            raise HTTPException(status_code=404, detail="sample not found") from None
        except FileNotFoundError:
            raise HTTPException(status_code=404,
                                detail=f"{kind} media is not ready") from None
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream",
                            filename=None)

    @app.post("/api/import")
    async def import_video(request: Request, name: str = Query(default="video.mp4")):
        # Raw-bytes upload keeps the endpoint free of python-multipart.
        content_type = (request.headers.get("content-type") or "").split(";")[0]
        if not content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="expected a video/* body")
        original_name = Path(name.replace("\\", "/")).name or "video.mp4"
        if not original_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            raise HTTPException(status_code=400, detail="unsupported video format")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty upload body")
        upload_dir = settings.dataset_root / "videos" / "imported" / ".uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        source = upload_dir / f"{token}.mp4"
        source.write_bytes(body)
        job_id = jobs.submit(source, original_name)
        return {"job_id": job_id}

    @app.get("/api/import/{job_id}")
    def import_status(job_id: str):
        try:
            return jobs.status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown import job") from None

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str):
        root: Path = settings.frontend_root
        requested = root / frontend_path
        cache_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
        if frontend_path and requested.is_file() and _path_is_within(requested, root):
            return FileResponse(requested, headers=cache_headers)
        index = root / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="frontend is not built")
        return FileResponse(index, headers=cache_headers)

    return app


app = create_app()


def main() -> None:
    """Run the packaged demo web service."""
    import uvicorn

    uvicorn.run(
        "dahua_cup.demo.app:app",
        host=os.environ.get("DAHUA_DEMO_HOST", "0.0.0.0"),
        port=int(os.environ.get("DAHUA_DEMO_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    main()
