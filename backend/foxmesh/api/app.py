"""HTTP API (FastAPI)."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image
from pydantic import BaseModel, Field

from .. import __version__
from ..benchmark import DEFAULT_CASES, BenchmarkCase, BenchmarkRun
from ..classifier import analyze
from ..jobs import UnknownUpload
from ..pipeline import Job, JobStatus
from ..router import NoCandidateError
from ..schemas import JobRequest, ModelSpec, ModelView, Profile, RoutingDecision, RoutingPolicy, TaskSpec
from ..services import Services, build_services

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".blend"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
IMMUTABLE = "public, max-age=31536000, immutable"


class SubmitResponse(BaseModel):
    job: Job
    deduplicated: bool


class AnalyzeResponse(BaseModel):
    spec: TaskSpec
    routing: RoutingDecision
    error: str | None = None


class ModelUpdate(BaseModel):
    enabled: bool | None = None
    version: str | None = None
    est_runtime_s: float | None = Field(None, gt=0)
    vram_gb: float | None = Field(None, ge=0)
    config: dict[str, Any] | None = None


class BenchmarkRequest(BaseModel):
    model_ids: list[str] | None = None
    case_ids: list[str] | None = None
    repeats: int = Field(1, ge=1, le=20)


class UploadResponse(BaseModel):
    id: str
    filename: str
    size: int


def create_app(services: Services | None = None) -> FastAPI:
    services = services or build_services()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await services.jobs.start()
        yield
        await services.benchmarks.stop()
        await services.jobs.stop()
        services.store.close()

    app = FastAPI(title="FoxMesh", version=__version__, lifespan=lifespan)
    app.state.services = services
    origins = os.environ.get("FOXMESH_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    def _job_or_404(job_id: str) -> Job:
        job = services.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    # -- system ----------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        s = services.settings
        return {
            "gpu": services.gpu.snapshot(),
            "blender": {"available": s.blender_available, "executable": s.blender_executable,
                        "mode": s.blender_mode},
            "queue": {"queued": services.jobs.queue_size, "running": services.jobs.running_count,
                      "workers": services.jobs.concurrency},
            "adapters": services.adapters.keys(),
        }

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        models = services.registry.views()
        return {
            "jobs": services.jobs.counts(),
            "recent_jobs": [_job_summary(j) for j in services.jobs.list(10)],
            "recent_errors": [_job_summary(j) for j in services.jobs.list(10, JobStatus.FAILED)],
            "models": [{
                "id": v.spec.id, "name": v.spec.name, "available": v.available,
                "runs": v.stats.runs, "failure_rate": v.stats.recent_failure_rate,
                "mean_runtime_s": v.stats.mean_runtime_s,
            } for v in models],
            "gpu": services.gpu.snapshot(),
        }

    # -- uploads -----------------------------------------------------------------

    @app.put("/api/uploads", response_model=UploadResponse)
    async def upload(request: Request, filename: str = Query(...)) -> UploadResponse:
        ext = Path(filename).suffix.lower()
        if ext not in UPLOAD_EXTENSIONS:
            raise HTTPException(400, f"unsupported file type {ext or '(none)'}")
        upload_id = uuid.uuid4().hex
        path = services.settings.uploads_dir / f"{upload_id}{ext}"
        size = 0
        with path.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, "file too large")
                fh.write(chunk)
        if size == 0:
            path.unlink(missing_ok=True)
            raise HTTPException(400, "empty upload")
        if ext in IMAGE_EXTENSIONS:
            # Reject undecodable images now rather than as a failed job later.
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                path.unlink(missing_ok=True)
                raise HTTPException(400, f"not a valid image: {exc}") from exc
        return UploadResponse(id=upload_id, filename=filename, size=size)

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str) -> FileResponse:
        """Serve an uploaded reference image (for previews)."""
        try:
            path = services.jobs.resolve_upload(upload_id)
        except UnknownUpload as exc:
            raise HTTPException(404, "upload not found") from exc
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(404, "upload not found")
        return FileResponse(path, headers={"Cache-Control": IMMUTABLE})

    # -- analysis & jobs -----------------------------------------------------------

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze_request(body: JobRequest) -> AnalyzeResponse:
        """Dry run: show how a request would be classified and routed, without running it."""
        try:
            spec = analyze(
                body,
                image_path=services.jobs.resolve_upload(body.image_id),
                input_mesh_path=services.jobs.resolve_upload(body.input_mesh_id),
                commercial_use=services.settings.commercial_use,
            )
        except UnknownUpload as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            return AnalyzeResponse(spec=spec, routing=services.router.route(spec))
        except NoCandidateError as exc:
            return AnalyzeResponse(spec=spec, routing=exc.decision, error=str(exc))

    # Handlers that touch the job queue or asyncio tasks are ``async def`` so they run on
    # the event loop; plain ``def`` handlers run in a thread pool, where asyncio objects
    # must not be used.
    @app.post("/api/jobs", response_model=SubmitResponse, status_code=202)
    async def submit(body: JobRequest) -> SubmitResponse:
        if not body.prompt.strip() and not body.image_id and not body.input_mesh_id:
            raise HTTPException(400, "provide a prompt, an image or an input 3D file")
        try:
            job, dedup = services.jobs.submit(body)
        except UnknownUpload as exc:
            raise HTTPException(400, str(exc)) from exc
        return SubmitResponse(job=job, deduplicated=dedup)

    @app.get("/api/jobs")
    def list_jobs(
        limit: int = Query(50, ge=1, le=500), status: JobStatus | None = None,
    ) -> list[dict[str, Any]]:
        return [_job_summary(j) for j in services.jobs.list(limit, status)]

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return _job_or_404(job_id)

    @app.post("/api/jobs/{job_id}/cancel", response_model=Job)
    async def cancel_job(job_id: str) -> Job:
        _job_or_404(job_id)
        return services.jobs.cancel(job_id)

    @app.post("/api/maintenance/cleanup")
    def cleanup() -> dict[str, int]:
        """Apply the retention policy now (it also runs hourly)."""
        return services.jobs.cleanup()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> FileResponse:
        job = _job_or_404(job_id)
        if job.result is None:
            raise HTTPException(404, "job has no result")
        allowed = set(job.result.files.values()) | {job.result.thumbnail}
        if name not in allowed:
            raise HTTPException(404, "file not found")
        path = Path(job.result.result_dir) / name
        if not path.is_file():
            raise HTTPException(404, "file not found")
        # A job's results never change after it finishes, so browsers and proxies may keep them.
        return FileResponse(path, filename=name, headers={"Cache-Control": IMMUTABLE})

    # -- models (execution engine management) ---------------------------------------

    @app.get("/api/models", response_model=list[ModelView])
    def list_models() -> list[ModelView]:
        return services.registry.views()

    @app.get("/api/models/{model_id}", response_model=ModelView)
    def get_model(model_id: str) -> ModelView:
        view = services.registry.view(model_id)
        if view is None:
            raise HTTPException(404, "model not found")
        return view

    @app.post("/api/models", response_model=ModelView, status_code=201)
    def register_model(spec: ModelSpec) -> ModelView:
        if services.registry.get(spec.id):
            raise HTTPException(409, "model id already registered")
        services.registry.register(spec)
        return services.registry.view(spec.id)

    @app.patch("/api/models/{model_id}", response_model=ModelView)
    def update_model(model_id: str, body: ModelUpdate) -> ModelView:
        try:
            services.registry.update(model_id, **body.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(404, "model not found") from exc
        return services.registry.view(model_id)

    @app.delete("/api/models/{model_id}/stats", response_model=ModelView)
    def reset_model_stats(model_id: str) -> ModelView:
        if services.registry.get(model_id) is None:
            raise HTTPException(404, "model not found")
        services.registry.stats.reset(model_id)
        return services.registry.view(model_id)

    # -- benchmarks (model performance evaluation) -------------------------------------

    @app.get("/api/benchmarks/cases", response_model=list[BenchmarkCase])
    def benchmark_cases() -> list[BenchmarkCase]:
        return DEFAULT_CASES

    @app.post("/api/benchmarks", response_model=BenchmarkRun, status_code=202)
    async def start_benchmark(body: BenchmarkRequest) -> BenchmarkRun:
        try:
            return services.benchmarks.start(body.model_ids, body.case_ids, body.repeats)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/benchmarks", response_model=list[BenchmarkRun])
    def list_benchmarks(limit: int = Query(20, ge=1, le=100)) -> list[BenchmarkRun]:
        return services.benchmarks.list(limit)

    @app.get("/api/benchmarks/{run_id}", response_model=BenchmarkRun)
    def get_benchmark(run_id: str) -> BenchmarkRun:
        run = services.benchmarks.get(run_id)
        if run is None:
            raise HTTPException(404, "benchmark not found")
        return run

    # -- monitoring ----------------------------------------------------------------------

    @app.get("/api/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus text format: jobs, queue, per-model runs/failures/latency, GPU memory."""
        lines: list[str] = []

        def metric(name: str, kind: str, help_: str, samples: list[tuple[dict[str, str], float]]):
            lines.append(f"# HELP foxmesh_{name} {help_}")
            lines.append(f"# TYPE foxmesh_{name} {kind}")
            for labels, value in samples:
                label = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"foxmesh_{name}{{{label}}} {value}" if label else f"foxmesh_{name} {value}")

        metric("jobs", "gauge", "Stored jobs by status.",
               [({"status": k}, v) for k, v in services.jobs.counts().items()])
        metric("queue_jobs", "gauge", "Jobs waiting in this process.", [({}, services.jobs.queue_size)])
        metric("running_jobs", "gauge", "Jobs running in this process.", [({}, services.jobs.running_count)])
        views = services.registry.views()
        metric("model_available", "gauge", "1 if the model can run now.",
               [({"model": v.spec.id}, int(v.available)) for v in views])
        metric("model_runs_total", "counter", "Recorded model runs.",
               [({"model": v.spec.id}, v.stats.runs) for v in views])
        metric("model_failures_total", "counter", "Recorded model failures.",
               [({"model": v.spec.id}, v.stats.failures) for v in views])
        metric("model_runtime_seconds", "gauge", "Mean measured generation time.",
               [({"model": v.spec.id}, v.stats.mean_runtime_s) for v in views
                if v.stats.mean_runtime_s is not None])
        metric("model_validation_score", "gauge", "Mean validation score.",
               [({"model": v.spec.id}, v.stats.mean_validation_score) for v in views
                if v.stats.mean_validation_score is not None])
        gpu = services.gpu.snapshot()
        metric("gpu_vram_total_gb", "gauge", "GPU memory.", [({}, gpu.total_vram_gb)])
        metric("gpu_vram_free_gb", "gauge", "Schedulable GPU memory.", [({}, gpu.free_vram_gb)])
        metric("gpu_vram_reserved_gb", "gauge", "GPU memory reserved by running jobs.",
               [({}, gpu.reserved_vram_gb)])
        metric("blender_available", "gauge", "1 if Blender post-processing is available.",
               [({}, int(services.settings.blender_available))])
        return "\n".join(lines) + "\n"

    # -- routing policies ------------------------------------------------------------

    @app.get("/api/policies", response_model=dict[Profile, RoutingPolicy])
    def list_policies() -> dict[Profile, RoutingPolicy]:
        return services.policies.all()

    @app.put("/api/policies/{profile}", response_model=RoutingPolicy)
    def set_policy(profile: Profile, policy: RoutingPolicy) -> RoutingPolicy:
        services.policies.set(profile, policy)
        return services.policies.get(profile)

    @app.delete("/api/policies/{profile}", response_model=RoutingPolicy)
    def reset_policy(profile: Profile) -> RoutingPolicy:
        services.policies.reset(profile)
        return services.policies.get(profile)

    return app


def _job_summary(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "prompt": job.spec.prompt,
        "task_type": job.spec.task_type,
        "profile": job.spec.profile,
        "model_id": job.result.model_id if job.result else (job.routing.selected if job.routing else None),
        "thumbnail": job.result.thumbnail if job.result else None,
        "error": job.error,
        "created_at": job.created_at,
        "duration_s": round(job.finished_at - job.started_at, 3)
        if job.finished_at and job.started_at else None,
    }
