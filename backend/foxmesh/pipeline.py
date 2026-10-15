"""End-to-end execution: route -> generate -> Blender post-process -> validate -> record.

Failures are handled with a bounded policy instead of open-ended agent loops:
transient errors (timeouts, busy servers) retry the same model a limited number
of times; any other failure, including a failed validation, moves on to the
next-ranked fallback model. The total number of attempts is capped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .adapters import AdapterRegistry, TransientError
from .blender import BlenderWorker, LitePostProcessor, PostProcessResult
from .gpu import GpuMonitor
from .registry import ModelRegistry
from .router import NoCandidateError, Router
from .schemas import JobRequest, OutputFormat, RoutingDecision, Runtime, TaskSpec
from .validation import ValidationResult, validate

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(str, Enum):
    QUEUED = "queued"
    ROUTING = "routing"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    VALIDATING = "validating"
    DONE = "done"


class JobEvent(BaseModel):
    ts: float
    stage: Stage
    message: str


class Attempt(BaseModel):
    model_id: str | None
    started_at: float
    generation_s: float | None = None
    postprocess_s: float | None = None
    ok: bool = False
    error: str | None = None
    transient: bool = False
    validation: ValidationResult | None = None


class JobResult(BaseModel):
    model_id: str | None
    engine: str
    steps: list[str]
    files: dict[str, str] = Field(description="format -> file name inside the result directory")
    thumbnail: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    validation: ValidationResult
    warnings: list[str] = Field(default_factory=list)
    result_dir: str


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    stage: Stage = Stage.QUEUED
    request: JobRequest
    spec: TaskSpec
    fingerprint: str
    routing: RoutingDecision | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    result: JobResult | None = None
    error: str | None = None
    events: list[JobEvent] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def log(self, stage: Stage, message: str) -> None:
        self.stage = stage
        self.events.append(JobEvent(ts=time.time(), stage=stage, message=message))

    @property
    def finished(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class Pipeline:
    def __init__(
        self,
        registry: ModelRegistry,
        adapters: AdapterRegistry,
        router: Router,
        gpu: GpuMonitor,
        postprocessor: BlenderWorker | LitePostProcessor,
        jobs_dir: Path,
        max_attempts: int = 3,
        transient_retries: int = 1,
        generation_timeout_s: float = 900.0,
        on_update: Callable[[Job], None] | None = None,
    ):
        self.registry = registry
        self.adapters = adapters
        self.router = router
        self.gpu = gpu
        self.postprocessor = postprocessor
        self.jobs_dir = jobs_dir
        self.max_attempts = max_attempts
        self.transient_retries = transient_retries
        self.generation_timeout_s = generation_timeout_s
        self._on_update = on_update or (lambda job: None)

    def _update(self, job: Job, stage: Stage, message: str) -> None:
        job.log(stage, message)
        self._on_update(job)

    async def run(self, job: Job) -> Job:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._update(job, Stage.ROUTING, "selecting execution path")
            try:
                job.routing = self.router.route(job.spec)
            except NoCandidateError as exc:
                job.routing = exc.decision
                raise
            if job.routing.requires_model:
                c = job.routing.candidates[0]
                self._update(job, Stage.ROUTING,
                             f"selected {c.model_name} (score {c.scores.total:.2f}, "
                             f"~{c.est_runtime_s:.0f}s) in {job.routing.decided_in_ms:.1f} ms")
                await self._run_generative(job, job_dir)
            else:
                self._update(job, Stage.ROUTING, f"{job.spec.task_type.value}: Blender only")
                await self._run_blender_only(job, job_dir)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.error = "cancelled"
            self._update(job, job.stage, "cancelled")
            raise
        except Exception as exc:
            log.exception("job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            self._update(job, job.stage, f"failed: {exc}")
        finally:
            job.finished_at = time.time()
            (job_dir / "job.json").write_text(job.model_dump_json(indent=1))
        return job

    # -- generative path -----------------------------------------------------

    async def _run_generative(self, job: Job, job_dir: Path) -> None:
        assert job.routing is not None
        order = [job.routing.selected, *job.routing.fallbacks]
        for model_id in order:
            if len(job.attempts) >= self.max_attempts:
                break
            retries = 0
            while len(job.attempts) < self.max_attempts:
                attempt_dir = job_dir / f"attempt-{len(job.attempts) + 1}"
                outcome = await self._attempt(job, model_id, attempt_dir)
                if outcome == "ok":
                    return
                if outcome == "transient" and retries < self.transient_retries:
                    retries += 1
                    self._update(job, Stage.GENERATING, f"retrying {model_id} after transient error")
                    continue
                break
            if len(job.attempts) < self.max_attempts and model_id != order[-1]:
                self._update(job, Stage.GENERATING, f"{model_id} failed; trying fallback")
        errors = "; ".join(f"{a.model_id}: {a.error}" for a in job.attempts)
        raise RuntimeError(f"all {len(job.attempts)} attempts failed ({errors})")

    async def _attempt(self, job: Job, model_id: str, attempt_dir: Path) -> str:
        spec = self.registry.get(model_id)
        adapter = self.adapters.get(spec.adapter) if spec else None
        attempt = Attempt(model_id=model_id, started_at=time.time())
        job.attempts.append(attempt)
        if spec is None or adapter is None:
            attempt.error = "model or adapter no longer registered"
            return "failed"

        gen_dir = attempt_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        self._update(job, Stage.GENERATING, f"generating with {spec.name}")
        vram = spec.vram_gb if spec.runtime is Runtime.LOCAL_GPU else 0.0
        t0 = time.monotonic()
        try:
            async with self.gpu.reserve(vram, timeout_s=self.generation_timeout_s):
                generated = await asyncio.wait_for(
                    adapter.generate(job.spec, spec, gen_dir), self.generation_timeout_s
                )
        except (TransientError, asyncio.TimeoutError) as exc:
            attempt.generation_s = time.monotonic() - t0
            attempt.error = str(exc) or "generation timed out"
            attempt.transient = True
            self.registry.stats.record(model_id, ok=False, runtime_s=None, error=attempt.error,
                                       extra={"job_id": job.id, "stage": "generate"})
            return "transient"
        except Exception as exc:
            attempt.generation_s = time.monotonic() - t0
            attempt.error = f"{type(exc).__name__}: {exc}"
            self.registry.stats.record(model_id, ok=False, runtime_s=None, error=attempt.error,
                                       extra={"job_id": job.id, "stage": "generate"})
            return "failed"
        attempt.generation_s = round(time.monotonic() - t0, 3)

        post, validation = await self._postprocess_and_validate(job, generated.path, attempt_dir / "result")
        attempt.postprocess_s = round(post.duration_s, 3)
        attempt.validation = validation
        attempt.ok = validation.passed
        if not validation.passed:
            attempt.error = "validation failed: " + ", ".join(
                f"{c.name} ({c.detail})" for c in validation.failures
            )
        self.registry.stats.record(
            model_id, ok=validation.passed, runtime_s=attempt.generation_s,
            validation_score=validation.score, error=attempt.error,
            extra={"job_id": job.id, "profile": job.spec.profile.value,
                   "postprocess_s": attempt.postprocess_s},
        )
        if validation.passed:
            self._finish(job, model_id, post, validation)
            return "ok"
        return "failed"

    # -- Blender-only path ---------------------------------------------------

    async def _run_blender_only(self, job: Job, job_dir: Path) -> None:
        if not job.spec.input_mesh_path:
            raise ValueError(f"{job.spec.task_type.value} needs an input 3D file")
        attempt = Attempt(model_id=None, started_at=time.time())
        job.attempts.append(attempt)
        post, validation = await self._postprocess_and_validate(
            job, Path(job.spec.input_mesh_path), job_dir / "result"
        )
        attempt.postprocess_s = round(post.duration_s, 3)
        attempt.validation = validation
        attempt.ok = validation.passed
        if not validation.passed:
            attempt.error = ", ".join(f"{c.name} ({c.detail})" for c in validation.failures)
            raise RuntimeError(f"validation failed: {attempt.error}")
        self._finish(job, None, post, validation)

    # -- shared ----------------------------------------------------------------

    async def _postprocess_and_validate(
        self, job: Job, source: Path, outdir: Path
    ) -> tuple[PostProcessResult, ValidationResult]:
        self._update(job, Stage.POSTPROCESSING, f"post-processing with {self.postprocessor.__class__.__name__}")
        post = await self.postprocessor.process(job.spec, source, outdir)
        self._update(job, Stage.VALIDATING, f"validating ({', '.join(post.steps)})")
        validation = validate(job.spec, post)
        return post, validation

    def _finish(self, job: Job, model_id: str | None, post: PostProcessResult,
                validation: ValidationResult) -> None:
        result_dir = post.outdir or next(iter(post.outputs.values())).parent
        job.result = JobResult(
            model_id=model_id,
            engine=post.engine,
            steps=post.steps,
            files={fmt.value: path.name for fmt, path in post.outputs.items()
                   if isinstance(fmt, OutputFormat)},
            thumbnail=post.thumbnail.name if post.thumbnail else None,
            stats=post.stats,
            validation=validation,
            warnings=post.warnings + [f"{c.name}: {c.detail}" for c in validation.checks
                                      if not c.passed and c.severity == "warning"],
            result_dir=str(result_dir),
        )
        job.status = JobStatus.SUCCEEDED
        self._update(job, Stage.DONE, f"done (validation score {validation.score:.2f})")
