"""FastAPI entry point for remote visualization and human review."""

from __future__ import annotations

import json
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dahua_cup.semantic_teacher.schemas import LABELS

from .baseline import Campus6Baseline
from .config import Settings, path_is_within
from .jobs import ACTIONS, JobManager
from .store import REVIEW_LABELS, ReviewStore


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    final_label: str
    reason_code: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=2000)


class UndoRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=80)


class JobRequest(BaseModel):
    action: str


class GPUSettingsRequest(BaseModel):
    pose_gpu_ids: List[int]
    student_gpu_ids: List[int]


def _not_found(kind: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} not found: {identifier}")


def _sample_payload(app: FastAPI, sample: dict) -> dict:
    sample = dict(sample)
    sample_id = sample["sample_id"]
    manager: JobManager = app.state.jobs
    sample["artifacts"] = manager.artifact_status(sample_id)
    # Raw RGB paths are strictly server-side implementation details.  The
    # browser only receives the derived pose video and inference artefacts.
    sample.pop("video_path", None)
    sample["media"] = {
        "pose": f"/api/samples/{sample_id}/media/pose",
    }
    sample["prediction"] = manager.prediction(sample_id)
    sample["teacher"] = manager.teacher_state(sample_id)
    return sample


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        store = ReviewStore(settings.database_path)
        annotations = settings.baseline_annotations()
        baseline = (
            Campus6Baseline(annotations, settings.baseline_predictions())
            if annotations is not None else None
        )
        app.state.settings = settings
        app.state.store = store
        app.state.baseline = baseline
        app.state.baseline_import = (
            store.import_baseline_records(baseline.records())
            if baseline is not None and baseline.available
            else {"inserted": 0, "existing": 0}
        )
        app.state.jobs = JobManager(settings, store, baseline)
        app.state.manifest_import = store.import_manifest(settings.manifest_path)
        app.state.jobs.start_continuous_pipeline()
        yield
        app.state.jobs.stop_continuous_pipeline()
        app.state.jobs.executor.shutdown(wait=False)

    app = FastAPI(
        title="校园行为语义理解平台",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "dahua-behavior-web"}

    @app.get("/api/capabilities")
    def capabilities(request: Request):
        return request.app.state.jobs.capability_state()

    @app.get("/api/system")
    def system(request: Request):
        current: Settings = request.app.state.settings
        return {
            "repository_root": str(current.repository_root),
            "data_root": str(current.data_root),
            "runtime_root": str(current.runtime_root),
            "source_root": str(current.source_root),
            "manifest_path": str(current.manifest_path),
            "database_path": str(current.database_path),
            "artifact_root": str(current.artifact_root),
            "active_student_checkpoint": str(
                current.resolve_student_checkpoint() or ""
            ),
            "student_runtime": {
                "name": "M1FKD INT8",
                "role": "Web inference deployment model",
                "checkpoint": str(current.resolve_student_checkpoint() or ""),
                "execution": "portable INT8 weights loaded through the deployment graph",
                "training_baseline": str(
                    current.training_baseline_checkpoint() or ""
                ),
            },
            "rtmpose_joint_score_threshold": current.joint_score_threshold,
            "manifest_import": request.app.state.manifest_import,
            "baseline_import": request.app.state.baseline_import,
            "labels": list(LABELS),
            "teacher_routing_config": (
                str(current.teacher_routing_config)
                if current.teacher_routing_config else None
            ),
            "teacher_trigger_confidence": current.teacher_trigger_confidence,
            "teacher_trigger_margin": current.teacher_trigger_margin,
            "pose_quality_threshold": current.pose_quality_threshold,
            "student_instability_threshold": (
                current.student_instability_threshold
            ),
            "teacher_conflict_confidence": (
                current.teacher_conflict_confidence
            ),
            "teacher_review_priorities": {
                "student_teacher_conflict": (
                    current.priority_teacher_conflict
                ),
                "pose_quality_failure": current.priority_pose_quality,
                "student_instability": (
                    current.priority_student_instability
                ),
                "teacher_requested_review": (
                    current.priority_teacher_review
                ),
                "uncertain_teacher_unavailable": (
                    current.priority_teacher_unavailable
                ),
            },
            "max_workers": current.max_workers,
            "continuous_pipeline": request.app.state.jobs.continuous_status(),
        }

    @app.get("/api/gpus")
    def gpu_state(request: Request):
        return request.app.state.jobs.gpus.state()

    @app.put("/api/gpus")
    def update_gpu_state(value: GPUSettingsRequest, request: Request):
        try:
            return request.app.state.jobs.gpus.update(
                value.pose_gpu_ids, value.student_gpu_ids
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/dashboard")
    def dashboard(request: Request):
        return request.app.state.store.dashboard()

    @app.get("/api/samples")
    def samples(
        request: Request,
        status: Optional[str] = None,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        offset: int = 0,
        limit: int = Query(default=50, ge=1, le=200),
    ):
        result = request.app.state.store.list_samples(
            status=status, dataset=dataset, query=query, offset=offset, limit=limit
        )
        result["items"] = [_sample_payload(request.app, item) for item in result["items"]]
        return result

    @app.get("/api/samples/{sample_id}")
    def sample_detail(sample_id: str, request: Request):
        try:
            sample = request.app.state.store.get_sample(sample_id)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        return _sample_payload(request.app, sample)

    @app.post("/api/samples/import-manifest")
    def import_manifest(request: Request):
        current: Settings = request.app.state.settings
        if not current.manifest_path.is_file():
            raise HTTPException(status_code=404, detail=f"manifest not found: {current.manifest_path}")
        return request.app.state.store.import_manifest(current.manifest_path)

    @app.post("/api/samples/{sample_id}/review")
    def submit_review(sample_id: str, value: ReviewRequest, request: Request):
        if value.final_label not in REVIEW_LABELS:
            raise HTTPException(status_code=422, detail="invalid final label")
        try:
            sample = request.app.state.store.submit_review(
                sample_id,
                value.reviewer,
                value.final_label,
                value.reason_code,
                value.note,
                student_snapshot=request.app.state.jobs.student_snapshot(sample_id),
                teacher_snapshot=request.app.state.jobs.teacher_snapshot(sample_id),
            )
        except KeyError:
            raise _not_found("sample", sample_id) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _sample_payload(request.app, sample)

    @app.post("/api/samples/{sample_id}/undo")
    def undo_review(sample_id: str, value: UndoRequest, request: Request):
        try:
            sample = request.app.state.store.undo_last_review(sample_id, value.actor)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _sample_payload(request.app, sample)

    @app.post("/api/reviews/undo-last")
    def undo_latest_review(value: UndoRequest, request: Request):
        try:
            sample = request.app.state.store.undo_latest_review(value.actor)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _sample_payload(request.app, sample)

    @app.post("/api/samples/{sample_id}/jobs")
    def create_job(sample_id: str, value: JobRequest, request: Request):
        if value.action not in ACTIONS:
            raise HTTPException(status_code=422, detail="unsupported action")
        try:
            return request.app.state.jobs.submit(sample_id, value.action)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str, request: Request):
        try:
            return request.app.state.store.get_job(job_id)
        except KeyError:
            raise _not_found("job", job_id) from None

    @app.get("/api/samples/{sample_id}/prediction")
    def prediction(sample_id: str, request: Request):
        try:
            request.app.state.store.get_sample(sample_id)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        value = request.app.state.jobs.prediction(sample_id)
        if value is None:
            raise HTTPException(status_code=404, detail="prediction is not ready")
        return value

    @app.get("/api/samples/{sample_id}/media/{kind}")
    def media(sample_id: str, kind: str, request: Request):
        try:
            sample = request.app.state.store.get_sample(sample_id)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        paths = request.app.state.jobs.artifacts(sample_id)
        if kind == "pose":
            path = request.app.state.jobs.pose_video(sample_id)
        elif kind in {"feature", "prediction"}:
            path = paths[kind]
        else:
            raise HTTPException(status_code=404, detail="unknown media kind")
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail=f"{kind} artifact is not ready")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream", filename=None)

    @app.post("/api/datasets/export")
    def export_dataset(request: Request):
        output = request.app.state.settings.runtime_root / "exports" / "manual_label_manifest.reviewed.csv"
        path = request.app.state.store.export_manifest(output)
        return {"path": str(path), "download_url": "/api/datasets/export/download"}

    @app.get("/api/datasets/export/download")
    def download_export(request: Request):
        path = request.app.state.settings.runtime_root / "exports" / "manual_label_manifest.reviewed.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="export has not been generated")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @app.get("/api/events")
    def events(request: Request, limit: int = Query(default=50, ge=1, le=200)):
        return {"items": request.app.state.store.recent_events(limit)}

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str, request: Request):
        root: Path = request.app.state.settings.frontend_root
        requested = root / frontend_path
        cache_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
        if frontend_path and requested.is_file() and path_is_within(requested, root):
            return FileResponse(requested, headers=cache_headers)
        index = root / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="frontend is not built")
        return FileResponse(index, headers=cache_headers)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "dahua_cup.backend.app:app",
        host=os.environ.get("DAHUA_VIS_HOST", "0.0.0.0"),
        port=int(os.environ.get("DAHUA_VIS_PORT", "8000")),
        reload=False,
    )
