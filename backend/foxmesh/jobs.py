"""Asynchronous job queue with duplicate detection and cancellation."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from .classifier import analyze
from .pipeline import Job, JobStatus, Pipeline
from .schemas import JobRequest

log = logging.getLogger(__name__)


class UnknownUpload(ValueError):
    pass


class JobManager:
    def __init__(
        self,
        pipeline: Pipeline,
        uploads_dir: Path,
        concurrency: int = 2,
        dedup_ttl_s: float = 3600.0,
        commercial_use: bool = True,
        max_history: int = 1000,
    ):
        self.pipeline = pipeline
        self.uploads_dir = uploads_dir
        self.concurrency = concurrency
        self.dedup_ttl_s = dedup_ttl_s
        self.commercial_use = commercial_use
        self.max_history = max_history
        self._jobs: dict[str, Job] = {}
        self._by_fingerprint: dict[str, str] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running: dict[str, asyncio.Task] = {}
        self._workers: list[asyncio.Task] = []

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]

    async def stop(self) -> None:
        for task in [*self._workers, *self._running.values()]:
            task.cancel()
        await asyncio.gather(*self._workers, *self._running.values(), return_exceptions=True)
        self._workers.clear()

    async def wait(self, job_id: str, timeout_s: float | None = None) -> Job:
        """Wait until a job finishes (used by the CLI and tests)."""
        async def _poll() -> Job:
            while not (job := self._jobs[job_id]).finished:
                await asyncio.sleep(0.05)
            return job
        return await asyncio.wait_for(_poll(), timeout_s)

    # -- submission ------------------------------------------------------------

    def resolve_upload(self, upload_id: str | None) -> str | None:
        if not upload_id:
            return None
        if not upload_id.isalnum():
            raise UnknownUpload(f"invalid upload id {upload_id!r}")
        matches = [p for p in self.uploads_dir.glob(f"{upload_id}.*") if p.is_file()]
        if not matches:
            raise UnknownUpload(f"unknown upload id {upload_id!r}")
        return str(matches[0])

    def submit(self, request: JobRequest) -> tuple[Job, bool]:
        """Queue a request. Returns ``(job, deduplicated)``.

        An identical request that is queued, running, or finished successfully
        within the dedup window returns the existing job instead of re-running.
        """
        spec = analyze(
            request,
            image_path=self.resolve_upload(request.image_id),
            input_mesh_path=self.resolve_upload(request.input_mesh_id),
            commercial_use=self.commercial_use,
        )
        fingerprint = spec.fingerprint()
        existing_id = self._by_fingerprint.get(fingerprint)
        if existing_id and (existing := self._jobs.get(existing_id)):
            fresh = time.time() - (existing.finished_at or existing.created_at) < self.dedup_ttl_s
            if not existing.finished or (existing.status is JobStatus.SUCCEEDED and fresh):
                return existing, True

        job = Job(id=uuid.uuid4().hex[:12], request=request, spec=spec, fingerprint=fingerprint)
        job.log(job.stage, "queued: " + ", ".join(spec.matched_rules))
        self._jobs[job.id] = job
        self._by_fingerprint[fingerprint] = job.id
        self._queue.put_nowait(job.id)
        self._trim_history()
        return job, False

    def cancel(self, job_id: str) -> Job:
        job = self._jobs[job_id]
        if job.finished:
            return job
        if task := self._running.get(job_id):
            task.cancel()
        else:
            job.status = JobStatus.CANCELLED
            job.error = "cancelled"
            job.finished_at = time.time()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: -j.created_at)[:limit]

    @property
    def queue_size(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status is JobStatus.QUEUED)

    @property
    def running_count(self) -> int:
        return len(self._running)

    # -- internals -------------------------------------------------------------

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            try:
                if job is None or job.status is not JobStatus.QUEUED:
                    continue  # cancelled while waiting
                task = asyncio.create_task(self.pipeline.run(job))
                self._running[job_id] = task
                try:
                    await task
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise  # the worker itself is being stopped
                    log.info("job %s cancelled", job_id)
                finally:
                    self._running.pop(job_id, None)
            finally:
                self._queue.task_done()

    def _trim_history(self) -> None:
        if len(self._jobs) <= self.max_history:
            return
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.created_at)
        for job in finished[: len(self._jobs) - self.max_history]:
            self._jobs.pop(job.id, None)
            if self._by_fingerprint.get(job.fingerprint) == job.id:
                self._by_fingerprint.pop(job.fingerprint, None)
