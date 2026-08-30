"""Standalone Web service for inspecting trained Group STN checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from dahua_cup.stn.inference import (
    STNInferenceOptions,
    STNVideoRunner,
    checkpoint_fingerprint,
    load_teacher_records,
    summarize_record,
    teacher_record_fingerprint,
)


@dataclass(frozen=True)
class STNVisualizationSettings:
    checkpoint: Path
    teacher_jsonl: Path
    output_root: Path
    frontend_root: Path
    host: str = "0.0.0.0"
    port: int = 8010
    device: str = "cuda:0"
    batch_size: int = 4
    window_step: int = 8
    presence_threshold: float = 0.5
    overlay_width: int = 960
    crop_size: int = 640
    ffmpeg: str = "ffmpeg"
    codec: str = "libopenh264"
    amp: bool = True

    @classmethod
    def from_env(cls) -> "STNVisualizationSettings":
        module_root = Path(__file__).resolve().parent
        data_root = Path(
            os.environ.get(
                "DAHUA_DATA_ROOT",
                "/workspace/data/xzz_data/DAHUA",
            )
        ).expanduser()
        return cls(
            checkpoint=Path(
                os.environ.get(
                    "DAHUA_STN_CHECKPOINT",
                    data_root / "models/stn/set_aware_group/best.pt",
                )
            ).expanduser(),
            teacher_jsonl=Path(
                os.environ.get(
                    "DAHUA_STN_TEACHER_JSONL",
                    data_root / "datasets/stn/yolo_tracks.jsonl",
                )
            ).expanduser(),
            output_root=Path(
                os.environ.get(
                    "DAHUA_STN_VIS_OUTPUT",
                    data_root / "runtime/stn_visualization",
                )
            ).expanduser(),
            frontend_root=module_root / "web",
            host=os.environ.get("DAHUA_STN_VIS_HOST", "0.0.0.0"),
            port=int(os.environ.get("DAHUA_STN_VIS_PORT", "8010")),
            device=os.environ.get("DAHUA_STN_DEVICE", "cuda:0"),
            batch_size=int(os.environ.get("DAHUA_STN_BATCH_SIZE", "4")),
            window_step=int(os.environ.get("DAHUA_STN_WINDOW_STEP", "8")),
            presence_threshold=float(
                os.environ.get("DAHUA_STN_PRESENCE_THRESHOLD", "0.5")
            ),
            overlay_width=int(
                os.environ.get("DAHUA_STN_OVERLAY_WIDTH", "960")
            ),
            crop_size=int(os.environ.get("DAHUA_STN_CROP_SIZE", "640")),
            ffmpeg=os.environ.get("DAHUA_FFMPEG", "ffmpeg"),
            codec=os.environ.get("DAHUA_PREVIEW_CODEC", "libopenh264"),
            amp=os.environ.get("DAHUA_STN_AMP", "1").strip().lower()
            not in {"0", "false", "no"},
        )

    def validate(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"STN checkpoint not found: {self.checkpoint}"
            )
        if not self.teacher_jsonl.is_file():
            raise FileNotFoundError(
                f"STN teacher JSONL not found: {self.teacher_jsonl}"
            )
        if not self.frontend_root.is_dir():
            raise FileNotFoundError(
                f"STN frontend not found: {self.frontend_root}"
            )
        if self.port < 1 or self.port > 65535:
            raise ValueError("STN visualization port is invalid")
        if self.batch_size < 1 or self.window_step < 1:
            raise ValueError("STN batch size and window step must be positive")
        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError("STN presence threshold must be in [0,1]")


class VisualizationManager:
    """Own teacher records, one GPU runner and serial visualization jobs."""

    def __init__(self, settings: STNVisualizationSettings) -> None:
        settings.validate()
        settings.output_root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        records = load_teacher_records(settings.teacher_jsonl)
        self.records = {
            str(record["sample_id"]): record for record in records
        }
        self.runner = STNVideoRunner(
            settings.checkpoint,
            settings.output_root,
            STNInferenceOptions(
                device=settings.device,
                batch_size=settings.batch_size,
                window_step=settings.window_step,
                presence_threshold=settings.presence_threshold,
                overlay_width=settings.overlay_width,
                crop_size=settings.crop_size,
                ffmpeg=settings.ffmpeg,
                codec=settings.codec,
                amp=settings.amp,
            ),
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stn-visualization"
        )
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def cached_result(self, sample_id: str) -> Optional[dict]:
        record = self.records[sample_id]
        safe = "".join(
            character
            if character.isalnum() or character in "._-"
            else "_"
            for character in str(record["sample_id"])
        ).strip("._")[:180]
        result_path = self.settings.output_root / safe / "result.json"
        if not result_path.is_file():
            return None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if result.get("checkpoint_fingerprint") != checkpoint_fingerprint(
            self.settings.checkpoint
        ):
            return None
        if result.get(
            "teacher_record_fingerprint"
        ) != teacher_record_fingerprint(record):
            return None
        return result

    def sample_payload(self, sample_id: str) -> dict:
        if sample_id not in self.records:
            raise KeyError(sample_id)
        payload = summarize_record(self.records[sample_id])
        result = self.cached_result(sample_id)
        payload["visualization_ready"] = result is not None
        payload["metrics"] = result.get("metrics") if result else None
        payload["media"] = {
            "overlay": f"/api/samples/{sample_id}/media/overlay",
            "crop": f"/api/samples/{sample_id}/media/crop",
        }
        return payload

    def list_samples(
        self,
        *,
        query: Optional[str],
        dataset: Optional[str],
        people: Optional[int],
        limit: int,
        offset: int,
    ) -> dict:
        query_text = (query or "").strip().lower()
        dataset_text = (dataset or "").strip().lower()
        values = []
        for sample_id, record in self.records.items():
            summary = summarize_record(record)
            if query_text and query_text not in (
                sample_id + " " + summary["video_path"]
            ).lower():
                continue
            if (
                dataset_text
                and summary["source_dataset"].lower() != dataset_text
            ):
                continue
            if people is not None and summary["selected_people"] != people:
                continue
            result = self.cached_result(sample_id)
            summary["visualization_ready"] = result is not None
            summary["metrics"] = result.get("metrics") if result else None
            values.append(summary)
        values.sort(
            key=lambda item: (
                item["source_dataset"].lower(),
                item["sample_id"].lower(),
            )
        )
        return {
            "items": values[offset:offset + limit],
            "total": len(values),
            "offset": offset,
            "limit": limit,
        }

    def submit(self, sample_id: str, *, force: bool) -> dict:
        if sample_id not in self.records:
            raise KeyError(sample_id)
        with self.lock:
            for job in self.jobs.values():
                if (
                    job["sample_id"] == sample_id
                    and job["status"] in {"queued", "running"}
                ):
                    return dict(job)
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "sample_id": sample_id,
                "status": "queued",
                "message": "等待STN推理",
                "error": None,
                "result": None,
            }
            self.jobs[job_id] = job
        self.executor.submit(self._run, job_id, force)
        return dict(job)

    def _run(self, job_id: str, force: bool) -> None:
        with self.lock:
            self.jobs[job_id].update(
                status="running",
                message="正在进行滑窗推理并渲染视频",
            )
        try:
            sample_id = self.jobs[job_id]["sample_id"]
            result = self.runner.run(self.records[sample_id], force=force)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(
                    status="failed",
                    message="STN可视化失败",
                    error=str(exc),
                )
            return
        with self.lock:
            self.jobs[job_id].update(
                status="completed",
                message="STN可视化完成",
                result=result,
            )

    def job(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


def create_app(
    settings: Optional[STNVisualizationSettings] = None,
) -> FastAPI:
    settings = settings or STNVisualizationSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = VisualizationManager(settings)
        yield
        app.state.manager.shutdown()

    app = FastAPI(
        title="Set-aware Group STN 可视化检验台",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    def health(request: Request):
        manager: VisualizationManager = request.app.state.manager
        return {
            "status": "ok",
            "service": "dahua-stn-visualization",
            "samples": len(manager.records),
            "device": manager.settings.device,
        }

    @app.get("/api/config")
    def configuration(request: Request):
        current = request.app.state.manager.settings
        return {
            "checkpoint": str(current.checkpoint),
            "checkpoint_fingerprint": checkpoint_fingerprint(
                current.checkpoint
            ),
            "teacher_jsonl": str(current.teacher_jsonl),
            "output_root": str(current.output_root),
            "device": current.device,
            "batch_size": current.batch_size,
            "window_step": current.window_step,
            "presence_threshold": current.presence_threshold,
            "amp": current.amp,
        }

    @app.get("/api/samples")
    def samples(
        request: Request,
        query: Optional[str] = None,
        dataset: Optional[str] = None,
        people: Optional[int] = Query(default=None, ge=0, le=2),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        return request.app.state.manager.list_samples(
            query=query,
            dataset=dataset,
            people=people,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/samples/{sample_id}")
    def sample(sample_id: str, request: Request):
        try:
            return request.app.state.manager.sample_payload(sample_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"sample not found: {sample_id}"
            ) from None

    @app.post("/api/samples/{sample_id}/run")
    def run_sample(
        sample_id: str,
        request: Request,
        force: bool = False,
    ):
        try:
            return request.app.state.manager.submit(
                sample_id, force=force
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"sample not found: {sample_id}"
            ) from None

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        try:
            return request.app.state.manager.job(job_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"job not found: {job_id}"
            ) from None

    @app.get("/api/samples/{sample_id}/result")
    def result(sample_id: str, request: Request):
        manager: VisualizationManager = request.app.state.manager
        if sample_id not in manager.records:
            raise HTTPException(
                status_code=404, detail=f"sample not found: {sample_id}"
            )
        value = manager.cached_result(sample_id)
        if value is None:
            raise HTTPException(
                status_code=404, detail="visualization is not ready"
            )
        return value

    @app.get("/api/samples/{sample_id}/media/{kind}")
    def media(sample_id: str, kind: str, request: Request):
        manager: VisualizationManager = request.app.state.manager
        if sample_id not in manager.records:
            raise HTTPException(
                status_code=404, detail=f"sample not found: {sample_id}"
            )
        result = manager.cached_result(sample_id)
        if result is None:
            raise HTTPException(
                status_code=404, detail="visualization is not ready"
            )
        if kind not in {"overlay", "crop"}:
            raise HTTPException(status_code=404, detail="unknown media kind")
        path = Path(result["artifacts"][kind])
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail=f"{kind} video is not ready"
            )
        return FileResponse(path, media_type="video/mp4")

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str):
        requested = (settings.frontend_root / frontend_path).resolve()
        root = settings.frontend_root.resolve()
        cache_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        }
        if (
            frontend_path
            and requested.is_file()
            and (requested == root or root in requested.parents)
        ):
            return FileResponse(requested, headers=cache_headers)
        return FileResponse(
            settings.frontend_root / "index.html",
            headers=cache_headers,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    defaults = STNVisualizationSettings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(defaults.checkpoint))
    parser.add_argument(
        "--teacher-jsonl", default=str(defaults.teacher_jsonl)
    )
    parser.add_argument("--output-dir", default=str(defaults.output_root))
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--window-step", type=int, default=defaults.window_step
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=defaults.presence_threshold,
    )
    parser.add_argument(
        "--overlay-width", type=int, default=defaults.overlay_width
    )
    parser.add_argument("--crop-size", type=int, default=defaults.crop_size)
    parser.add_argument("--ffmpeg", default=defaults.ffmpeg)
    parser.add_argument("--codec", default=defaults.codec)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    settings = STNVisualizationSettings(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        teacher_jsonl=Path(args.teacher_jsonl).expanduser().resolve(),
        output_root=Path(args.output_dir).expanduser().resolve(),
        frontend_root=Path(__file__).resolve().parent / "web",
        host=args.host,
        port=args.port,
        device=args.device,
        batch_size=args.batch_size,
        window_step=args.window_step,
        presence_threshold=args.presence_threshold,
        overlay_width=args.overlay_width,
        crop_size=args.crop_size,
        ffmpeg=args.ffmpeg,
        codec=args.codec,
        amp=not args.no_amp,
    )
    settings.validate()
    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
