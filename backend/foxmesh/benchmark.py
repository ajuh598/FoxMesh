"""Model benchmarks: run a fixed test set against every available model.

Each (model, case) pair runs through the normal pipeline with that model forced,
one attempt and no fallback, so the numbers measure the model itself: success
rate, generation time and validation score. Results are recorded in the model
statistics the router uses, so selection is driven by measurements rather than
the descriptor's prior estimates.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import statistics
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .adapters import AdapterRegistry
from .blender import BlenderWorker, LitePostProcessor
from .classifier import analyze
from .gpu import GpuMonitor
from .pipeline import Job, Pipeline
from .registry import ModelRegistry
from .router import NoCandidateError, Router
from .schemas import JobRequest, Profile

log = logging.getLogger(__name__)


class BenchmarkCase(BaseModel):
    id: str
    prompt: str = ""
    image: str | None = Field(None, description="Built-in synthetic image name")
    profile: Profile | None = None


DEFAULT_CASES = [
    BenchmarkCase(id="text-game-sword", prompt="게임에 사용할 중세풍 검"),
    BenchmarkCase(id="text-print-cup", prompt="3D 프린팅용 8cm 컵"),
    BenchmarkCase(id="text-prototype-chair", prompt="빠르게 의자 시제품"),
    BenchmarkCase(id="text-hq-house", prompt="고품질 집"),
    BenchmarkCase(id="image-apple", image="apple"),
    BenchmarkCase(id="image-star-keychain", prompt="로고 키링", image="star"),
]


class BenchmarkResult(BaseModel):
    model_id: str
    case_id: str
    ok: bool
    generation_s: float | None = None
    postprocess_s: float | None = None
    validation_score: float | None = None
    error: str | None = None


class ModelSummary(BaseModel):
    model_id: str
    runs: int
    successes: int
    success_rate: float
    mean_generation_s: float | None
    p90_generation_s: float | None
    mean_validation_score: float | None
    skipped_cases: list[str] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    id: str
    status: str = "running"  # running | done | failed
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    model_ids: list[str]
    case_ids: list[str]
    repeats: int = 1
    progress: int = 0
    total: int = 0
    results: list[BenchmarkResult] = Field(default_factory=list)
    skipped: dict[str, list[str]] = Field(default_factory=dict)  # model -> case ids
    summary: list[ModelSummary] = Field(default_factory=list)
    error: str | None = None


def _synthetic_image(name: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "apple":
        img = Image.new("RGB", (480, 400), "white")
        d = ImageDraw.Draw(img)
        d.ellipse((120, 80, 360, 330), fill=(200, 30, 40))
        d.ellipse((220, 40, 280, 95), fill=(40, 160, 60))
        d.ellipse((180, 130, 210, 160), fill=(255, 200, 200))
    elif name == "star":
        img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        pts = [(200 + 180 * math.cos(math.pi / 2 + i * math.pi / 5) * (1 if i % 2 == 0 else 0.45),
                200 - 180 * math.sin(math.pi / 2 + i * math.pi / 5) * (1 if i % 2 == 0 else 0.45))
               for i in range(10)]
        ImageDraw.Draw(img).polygon(pts, fill=(250, 200, 30, 255))
    else:
        raise ValueError(f"unknown synthetic image {name!r}")
    img.save(path)
    return path


def summarize(run: BenchmarkRun) -> list[ModelSummary]:
    out = []
    for model_id in run.model_ids:
        rows = [r for r in run.results if r.model_id == model_id]
        times = sorted(r.generation_s for r in rows if r.ok and r.generation_s is not None)
        scores = [r.validation_score for r in rows if r.validation_score is not None]
        successes = sum(r.ok for r in rows)
        out.append(ModelSummary(
            model_id=model_id,
            runs=len(rows),
            successes=successes,
            success_rate=round(successes / len(rows), 3) if rows else 0.0,
            mean_generation_s=round(statistics.fmean(times), 3) if times else None,
            p90_generation_s=round(times[min(len(times) - 1, int(0.9 * len(times)))], 3) if times else None,
            mean_validation_score=round(statistics.fmean(scores), 3) if scores else None,
            skipped_cases=run.skipped.get(model_id, []),
        ))
    return out


class BenchmarkManager:
    def __init__(
        self,
        registry: ModelRegistry,
        adapters: AdapterRegistry,
        router: Router,
        gpu: GpuMonitor,
        postprocessor: BlenderWorker | LitePostProcessor,
        data_dir: Path,
        commercial_use: bool = True,
    ):
        self.registry = registry
        self.router = router
        self.dir = data_dir / "benchmarks"
        self.commercial_use = commercial_use
        self.pipeline = Pipeline(registry, adapters, router, gpu, postprocessor, self.dir / "work",
                                 max_attempts=1, transient_retries=0)
        self._task: asyncio.Task | None = None
        self._current: BenchmarkRun | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, model_ids: list[str] | None = None, case_ids: list[str] | None = None,
              repeats: int = 1) -> BenchmarkRun:
        if self.running:
            raise RuntimeError("a benchmark is already running")
        available = [v.spec.id for v in self.registry.views() if v.available]
        models = [m for m in (model_ids or available) if m in available]
        if not models:
            raise ValueError("no available models to benchmark")
        cases = [c for c in DEFAULT_CASES if not case_ids or c.id in case_ids]
        if not cases:
            raise ValueError("no matching benchmark cases")
        run = BenchmarkRun(id=uuid.uuid4().hex[:10], model_ids=models,
                           case_ids=[c.id for c in cases], repeats=repeats)
        self._current = run
        self._save(run)
        self._task = asyncio.create_task(self._run(run, cases))
        return run

    async def wait(self) -> BenchmarkRun | None:
        if self._task:
            await self._task
        return self._current

    async def stop(self) -> None:
        if self.running:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def get(self, run_id: str) -> BenchmarkRun | None:
        if self._current and self._current.id == run_id:
            return self._current
        path = self.dir / f"{run_id}.json"
        if not run_id.isalnum() or not path.exists():
            return None
        return BenchmarkRun.model_validate_json(path.read_text())

    def list(self, limit: int = 20) -> list[BenchmarkRun]:
        runs = [BenchmarkRun.model_validate_json(p.read_text())
                for p in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
        if self._current:
            runs = [self._current] + [r for r in runs if r.id != self._current.id]
        return runs[:limit]

    def _save(self, run: BenchmarkRun) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{run.id}.json").write_text(run.model_dump_json(indent=1))

    def _spec(self, case: BenchmarkCase, model_id: str):
        image = _synthetic_image(case.image, self.dir / "assets" / f"{case.image}.png") if case.image else None
        request = JobRequest(prompt=case.prompt, profile=case.profile, model_id=model_id)
        return request, analyze(request, image_path=str(image) if image else None,
                                commercial_use=self.commercial_use)

    async def _run(self, run: BenchmarkRun, cases: list[BenchmarkCase]) -> None:
        try:
            plan = []
            for model_id in run.model_ids:
                for case in cases:
                    request, spec = self._spec(case, model_id)
                    try:
                        self.router.route(spec)
                    except NoCandidateError:
                        run.skipped.setdefault(model_id, []).append(case.id)
                        continue
                    plan.extend([(model_id, case, request, spec)] * run.repeats)
            run.total = len(plan)
            self._save(run)
            for model_id, case, request, spec in plan:
                job = Job(id=f"bench-{uuid.uuid4().hex[:10]}", request=request, spec=spec,
                          fingerprint=spec.fingerprint())
                job = await self.pipeline.run(job)
                attempt = job.attempts[-1] if job.attempts else None
                run.results.append(BenchmarkResult(
                    model_id=model_id, case_id=case.id, ok=bool(attempt and attempt.ok),
                    generation_s=attempt.generation_s if attempt else None,
                    postprocess_s=attempt.postprocess_s if attempt else None,
                    validation_score=attempt.validation.score if attempt and attempt.validation else None,
                    error=(attempt.error if attempt else job.error),
                ))
                run.progress += 1
                run.summary = summarize(run)
                self._save(run)
                # Benchmark outputs are not kept; the numbers are.
                shutil.rmtree(self.pipeline.jobs_dir / job.id, ignore_errors=True)
            run.status = "done"
        except asyncio.CancelledError:
            run.status = "failed"
            run.error = "cancelled"
            raise
        except Exception as exc:
            log.exception("benchmark %s failed", run.id)
            run.status = "failed"
            run.error = str(exc)
        finally:
            run.finished_at = time.time()
            run.summary = summarize(run)
            self._save(run)
