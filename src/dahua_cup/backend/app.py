"""FastAPI entry point for remote visualization and human review."""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dahua_cup.semantic_teacher.schemas import LABELS

from .baseline import Campus6Baseline
from .config import Settings, path_is_within
from .hard_samples import evaluate_hard_sample
from .jobs import ACTIONS, JobManager
from .macro_parameters import (
    apply_values,
    save_macro_parameter_file,
    snapshot,
)
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
    teacher_gpu_ids: List[int] = Field(default_factory=list)
    teacher_auto: bool = True


class MacroParametersRequest(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)


class QwenRemoteRequest(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 22
    user: str = ""
    project_root: str = ""


class QwenApiRequest(BaseModel):
    """Non-secret settings for the hosted Qwen fallback.

    ``api_key`` is write-only.  The backend never returns it and stores it in
    a separate owner-restricted server-local file rather than JSON settings.
    """

    enabled: bool = False
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    model: str = "qwen3-vl-flash"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: int = Field(default=180, ge=10, le=900)
    max_attempts: int = Field(default=2, ge=1, le=5)
    max_frames: int = Field(default=8, ge=2, le=12)
    api_key: str = Field(default="", max_length=2048)
    clear_api_key: bool = False


class PoseBackendRequest(BaseModel):
    backend_id: str = Field(min_length=1, max_length=80)


UPLOAD_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"})
UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _save_uploaded_video(upload: UploadFile, destination: Path) -> int:
    """Persist an upload with a strict size limit and no client path trust."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("DAHUA_WEB_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
    size = 0
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    try:
        with temporary.open("wb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError("上传视频超过 {} MiB 限制".format(limit // 1024 // 1024))
                output.write(chunk)
        if size == 0:
            raise ValueError("上传文件为空")
        temporary.replace(destination)
        return size
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _not_found(kind: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} not found: {identifier}")


def _difficulty_checks(
    prediction: Optional[dict], teacher: Optional[dict], settings: Settings
) -> list[dict]:
    decision = evaluate_hard_sample(
        prediction,
        teacher,
        confidence_threshold=settings.teacher_trigger_confidence,
        conflict_confidence_threshold=settings.teacher_conflict_confidence,
    )
    return [
        {"code": item["code"], "label": item["title"], "status": item["status"]}
        for item in decision["conditions"]
    ]


def _sample_payload(
    app: FastAPI, sample: dict, *, include_semantic_graph: bool = False
) -> dict:
    sample = dict(sample)
    sample_id = sample["sample_id"]
    manager: JobManager = app.state.jobs
    sample["artifacts"] = manager.artifact_status(sample_id)
    # Raw RGB paths are strictly server-side implementation details.  The
    # browser only receives the derived pose video and inference artefacts.
    sample.pop("video_path", None)
    sample.pop("hard_score", None)
    sample.pop("quality_score", None)
    sample["media"] = {
        "pose": f"/api/samples/{sample_id}/media/pose",
    }
    prediction = manager.prediction(sample_id)
    teacher = manager.teacher_state(sample_id)
    sample["prediction"] = prediction
    if include_semantic_graph:
        semantic_graph = (
            ((prediction or {}).get("student_evidence") or {}).get(
                "semantic_graph"
            )
        )
        baseline: Optional[Campus6Baseline] = app.state.baseline
        if semantic_graph is None and baseline is not None and baseline.has(sample_id):
            semantic_graph = baseline.semantic_graph(sample_id)
        sample["semantic_graph"] = semantic_graph
    sample["teacher"] = teacher
    decision = evaluate_hard_sample(
        prediction,
        teacher,
        confidence_threshold=manager.settings.teacher_trigger_confidence,
        conflict_confidence_threshold=manager.settings.teacher_conflict_confidence,
    )
    checks = [
        {"code": item["code"], "label": item["title"], "status": item["status"]}
        for item in decision["conditions"]
    ]
    sample["difficulty_checks"] = checks
    sample["hard_decision"] = decision
    teacher_finished = (
        teacher.get("status") in {"completed", "failed"}
        and bool(teacher.get("generated_at"))
    )
    check_status = {item["code"]: item["status"] for item in checks}
    teacher_needed = any(
        check_status.get(code) == "yes" for code in ("C1", "C5")
    )
    teacher_completed = (
        teacher.get("status") == "completed" and bool(teacher.get("result"))
    )
    human_needed = teacher_completed and (
        bool((teacher.get("result") or {}).get("needs_review"))
        or check_status.get("C2") == "yes"
        or check_status.get("C3") == "yes"
    )
    human_completed = (
        bool(sample.get("reviewer"))
        and sample.get("reviewer") != "official_initial_annotation"
    )
    if teacher_needed and not teacher_completed:
        workflow_status = "waiting_teacher"
        workflow_reason = "hard_sample_waiting_teacher"
    elif human_needed and not human_completed:
        workflow_status = "waiting_human"
        workflow_reason = "teacher_result_requires_review"
    else:
        workflow_status = "complete"
        workflow_reason = "inference_chain_completed"
    sample["workflow_status"] = workflow_status
    sample["workflow_reason"] = workflow_reason
    sample["produced_at"] = (
        teacher.get("generated_at")
        if teacher_finished else (prediction or {}).get("generated_at")
    )
    return sample


def _sample_summary_payload(
    app: FastAPI, sample: dict, *, include_teacher: bool = True
) -> dict:
    """Return list-view metadata without loading per-sample artefacts."""
    manager: JobManager = app.state.jobs
    prediction = manager.prediction(sample["sample_id"])
    topk = list((prediction or {}).get("topk") or [])
    student_label = topk[0].get("label") if topk else None
    teacher_label = None
    if include_teacher:
        teacher = manager.teacher_state(
            sample["sample_id"], prediction=prediction
        )
        teacher_result = dict(teacher.get("result") or {})
        teacher_label = (
            teacher_result.get("label") or teacher_result.get("suggested_label")
        )
    return {
        "sample_id": sample["sample_id"],
        "source_dataset": sample.get("source_dataset", ""),
        "source_label": sample.get("source_label", ""),
        "suggested_coarse_label": sample.get("suggested_coarse_label", ""),
        "suggested_label": sample.get("suggested_label"),
        "student_label": student_label,
        "teacher_label": teacher_label,
        "status": sample.get("status", "pending"),
        "workflow_status": sample.get("workflow_status", "complete"),
        "priority": sample.get("priority", 0),
        "manual_label": sample.get("manual_label"),
        "reviewer": sample.get("reviewer"),
        "produced_at": sample.get("reviewed_at") or sample.get("updated_at"),
    }


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        store = ReviewStore(settings.database_path)
        app.state.recovered_jobs = store.recover_incomplete_jobs()
        app.state.redacted_teacher_failures = store.redact_verbose_teacher_failures()
        annotations = settings.baseline_annotations()
        baseline = (
            Campus6Baseline(
                annotations,
                settings.baseline_predictions(),
                review_temperature=settings.student_probability_temperature(),
                ffmpeg=str(settings.ffmpeg or "ffmpeg"),
                preview_codec=settings.preview_codec,
                preview_preset=settings.preview_preset,
                preview_bitrate=settings.preview_bitrate,
            )
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
        app.state.jobs.shutdown()

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
                "name": "M1KD QAT INT8 + Logits KD",
                "role": "Web inference deployment model",
                "checkpoint": str(current.resolve_student_checkpoint() or ""),
                "execution": "portable INT8 weights loaded through the deployment graph",
                "training_baseline": str(
                    current.training_baseline_checkpoint() or ""
                ),
            },
            "pose_backend": current.pose_backend_status(),
            "rtmpose_joint_score_threshold": current.joint_score_threshold,
            "manifest_import": request.app.state.manifest_import,
            "baseline_import": request.app.state.baseline_import,
            "labels": list(LABELS),
            "teacher_routing_config": (
                str(current.teacher_routing_config)
                if current.teacher_routing_config else None
            ),
            "teacher_trigger_confidence": current.teacher_trigger_confidence,
            "student_instability_threshold": (
                current.student_instability_threshold
            ),
            "teacher_conflict_confidence": (
                current.teacher_conflict_confidence
            ),
            "max_workers": current.max_workers,
            "continuous_pipeline": request.app.state.jobs.continuous_status(),
        }

    @app.get("/api/gpus")
    def gpu_state(request: Request):
        return request.app.state.jobs.gpus.state()

    @app.get("/api/models/status")
    def model_status(request: Request):
        return request.app.state.jobs.model_status()

    @app.get("/api/pose-backend")
    def pose_backend(request: Request):
        return request.app.state.settings.pose_backend_status()

    @app.put("/api/pose-backend")
    def update_pose_backend(value: PoseBackendRequest, request: Request):
        try:
            return request.app.state.settings.update_pose_backend(value.backend_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/training/status")
    def incremental_training_status(request: Request):
        return request.app.state.jobs.incremental_training_status()

    @app.post("/api/gcn/upload")
    async def upload_gcn_video(
        request: Request,
        video: UploadFile = File(...),
    ):
        """Upload a local RGB video and immediately start the local pipeline.

        The browser sends RGB only to this local Web service.  The service
        stores it below ``runtime/videos``, runs RTMDet/RTMPose and ProtoGCN
        locally, and the existing uncertainty gate alone may later send the
        derived skeleton artefacts to the configured remote Qwen teacher.
        """
        original_name = Path(video.filename or "video.mp4").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail="仅支持 MP4、AVI、MOV、MKV、WebM、MPEG 视频",
            )
        capability = request.app.state.jobs.local_inference_capability()
        required = ("pose_extraction", "campus6_inference")
        unavailable = [capability[name]["reason"] for name in required if not capability[name]["enabled"]]
        if unavailable:
            raise HTTPException(status_code=409, detail="；".join(unavailable))
        sample_id = "local_{}_{}".format(
            uuid.uuid4().hex[:12], uuid.uuid4().hex[:8]
        )
        destination = request.app.state.settings.runtime_root / "videos" / (sample_id + suffix)
        sample = None
        job = None
        try:
            size = await _save_uploaded_video(video, destination)
            sample = request.app.state.store.add_sample(
                sample_id,
                destination,
                source_dataset="local_upload",
                source_label=original_name,
            )
            job = request.app.state.jobs.submit(sample_id, "full")
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            if sample is not None and job is None:
                request.app.state.store.discard_unstarted_local_upload(sample_id)
            destination.unlink(missing_ok=True)
            raise
        return {
            "sample": _sample_payload(request.app, sample),
            "job": job,
            "upload": {
                "original_name": original_name,
                "size_bytes": size,
                "execution": "local_rtmpose_and_protogcn",
                "teacher": "remote_qwen_only_if_gate_triggers",
            },
        }

    @app.put("/api/gpus")
    def update_gpu_state(value: GPUSettingsRequest, request: Request):
        try:
            return request.app.state.jobs.gpus.update(
                value.pose_gpu_ids,
                value.student_gpu_ids,
                value.teacher_gpu_ids,
                teacher_auto=value.teacher_auto,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/qwen-remote")
    def qwen_remote(request: Request):
        return request.app.state.settings.qwen_remote()

    @app.put("/api/qwen-remote")
    def update_qwen_remote(value: QwenRemoteRequest, request: Request):
        try:
            return request.app.state.settings.update_qwen_remote(
                value.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/qwen-api")
    def qwen_api(request: Request):
        return request.app.state.settings.qwen_api_status()

    @app.put("/api/qwen-api")
    def update_qwen_api(value: QwenApiRequest, request: Request):
        try:
            return request.app.state.settings.update_qwen_api(value.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/qwen-remote/test")
    def test_qwen_remote(request: Request):
        try:
            request.app.state.jobs.test_qwen_remote_connection()
            return {"ok": True}
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=422, detail="Qwen 云端连接失败：{}".format(str(exc)[:240])) from exc

    @app.post("/api/qwen-remote/provision")
    def provision_qwen_remote(request: Request):
        try:
            summary = request.app.state.jobs.provision_qwen_remote()
            if summary.get("training_env") == "already_present":
                message = "云端代码与 Qwen 虚拟环境已初始化（训练环境已存在，跳过）"
            elif summary.get("training_env") == "created":
                message = "云端代码与 Qwen 虚拟环境已初始化（训练环境已自动创建）"
            else:
                message = "云端代码与 Qwen 虚拟环境已初始化"
            return {"ok": True, "message": message, **summary}
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=422, detail="Qwen 云端初始化失败：{}".format(str(exc)[:240])) from exc

    @app.get("/api/macro-parameters")
    def macro_parameters(request: Request):
        return snapshot(request.app.state.settings)

    @app.put("/api/macro-parameters")
    def update_macro_parameters(value: MacroParametersRequest, request: Request):
        current: Settings = request.app.state.settings
        try:
            cleaned = apply_values(current, value.values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        save_macro_parameter_file(current.macro_parameters_path, cleaned)
        # The baseline review probabilities soften the cached temperatures
        # in place, so keep its copy of the temperature aligned as well.
        baseline: Optional[Campus6Baseline] = request.app.state.baseline
        if baseline is not None:
            baseline.review_temperature = current.review_temperature
        # The incremental dataset holds a live copy of the batch threshold;
        # keep it aligned so readiness checks use the new value immediately.
        jobs: Optional[JobManager] = request.app.state.jobs
        if jobs is not None and "incremental_batch_size" in cleaned:
            jobs.incremental.batch_size = cleaned["incremental_batch_size"]
        return snapshot(current)

    @app.get("/api/dashboard")
    def dashboard(request: Request):
        return request.app.state.store.dashboard()

    @app.get("/api/samples")
    def samples(
        request: Request,
        status: Optional[str] = None,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        workflow_status: Optional[str] = None,
        offset: int = 0,
        limit: int = Query(default=50, ge=1, le=500),
        summary: bool = False,
        include_teacher: bool = True,
    ):
        result = request.app.state.store.list_samples(
            status=status,
            dataset=dataset,
            query=query,
            workflow_status=workflow_status,
            offset=offset,
            limit=limit,
            summary=summary,
        )
        if summary:
            result["items"] = [
                _sample_summary_payload(
                    request.app, item, include_teacher=include_teacher
                )
                for item in result["items"]
            ]
        else:
            result["items"] = [_sample_payload(request.app, item) for item in result["items"]]
        return result

    @app.get("/api/samples/{sample_id}")
    def sample_detail(sample_id: str, request: Request):
        try:
            sample = request.app.state.store.get_sample(sample_id)
        except KeyError:
            raise _not_found("sample", sample_id) from None
        return _sample_payload(request.app, sample, include_semantic_graph=True)

    @app.get("/api/hard-samples")
    def hard_samples(
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        result = request.app.state.jobs.hard_samples(
            offset=offset, limit=limit
        )
        current: Settings = request.app.state.settings
        for item in result["items"]:
            item.pop("hard_score", None)
            item.pop("quality_score", None)
            item["artifacts"] = request.app.state.jobs.artifact_status(
                item["sample_id"]
            )
            item["difficulty_checks"] = _difficulty_checks(
                item.get("prediction"), item.get("teacher"), current
            )
        return result

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
        payload = _sample_payload(request.app, sample)
        if sample.get("incremental_pool"):
            payload["incremental_enrollment"] = (
                request.app.state.jobs.enroll_reviewed_hard_sample(sample_id)
            )
        return payload

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
        if frontend_path and requested.is_file() and path_is_within(requested, root):
            cache_headers = (
                {"Cache-Control": "public, max-age=31536000, immutable"}
                if requested.suffix in {".css", ".js"}
                else {"Cache-Control": "no-cache"}
            )
            return FileResponse(requested, headers=cache_headers)
        index = root / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="frontend is not built")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()


def main() -> None:
    """Run the packaged Campus6 Web service."""
    import uvicorn

    uvicorn.run(
        "dahua_cup.backend.app:app",
        host=os.environ.get("DAHUA_VIS_HOST", "0.0.0.0"),
        port=int(os.environ.get("DAHUA_VIS_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
