"""Asynchronous job queue with durable storage, duplicate detection, recovery and cleanup."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from .classifier import analyze
from .pipeline import Job, JobStatus, Pipeline, Stage
from .schemas import JobRequest
from .store import JobStore, MemoryJobStore

log = logging.getLogger(__name__)

# A job interrupted by a restart is re-run this many times before it is failed,
# so a job that crashes the process cannot put it into a restart loop.
MAX_RECOVERIES = 1


class UnknownUpload(ValueError):
    pass


class JobManager:
    def __init__(
        self,
        pipeline: Pipeline,
        uploads_dir: Path,
        store: JobStore | None = None,
        concurrency: int = 2,
        dedup_ttl_s: float = 3600.0,
        commercial_use: bool = True,
        retention_s: float | None = None,
        cleanup_interval_s: float = 3600.0,
    ):
        self.pipeline = pipeline
        self.uploads_dir = uploads_dir
        self.store = store or MemoryJobStore()
        self.concurrency = concurrency
        self.dedup_ttl_s = dedup_ttl_s
        self.commercial_use = commercial_use
        self.retention_s = retention_s
        self.cleanup_interval_s = cleanup_interval_s
        # Queued and running jobs of this process; the store holds everything.
        self._active: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running: dict[str, asyncio.Task] = {}
        self._workers: list[asyncio.Task] = []
        self._cleaner: asyncio.Task | None = None
        pipeline.on_update = self.store.save

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        if self._workers:
            return
        recovered = self.recover()
        if recovered:
            log.info("re-queued %d job(s) interrupted by a restart", recovered)
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]
        if self.retention_s:
            self._cleaner = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        tasks = [*self._workers, *self._running.values()]
        if self._cleaner:
            tasks.append(self._cleaner)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._cleaner = None

    def recover(self) -> int:
        """Re-queue jobs left queued or running by a previous process."""
        count = 0
        for job in self.store.unfinished():
            if job.id in self._active:
                continue
            if job.status is JobStatus.RUNNING:
                job.recoveries += 1
                if job.recoveries > MAX_RECOVERIES:
                    job.status = JobStatus.FAILED
                    job.error = "interrupted by a restart too many times"
                    job.finished_at = time.time()
                    job.log(job.stage, job.error)
                    self.store.save(job)
                    continue
                job.status = JobStatus.QUEUED
                job.log(Stage.QUEUED, "re-queued after a restart")
            self.store.save(job)
            self._active[job.id] = job
            self._queue.put_nowait(job.id)
            count += 1
        return count

    async def wait(self, job_id: str, timeout_s: float | None = None) -> Job:
        """Wait until a job finishes (used by the CLI and tests)."""
        async def _poll() -> Job:
            while not (job := self.get(job_id)).finished:
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

    def _reusable(self, fingerprint: str) -> Job | None:
        for job in self._active.values():
            if job.fingerprint == fingerprint:
                return job
        job = self.store.latest_with_fingerprint(fingerprint)
        if job is None or job.status is not JobStatus.SUCCEEDED or job.result is None:
            return None
        if time.time() - (job.finished_at or job.created_at) >= self.dedup_ttl_s:
            return None
        result_dir = Path(job.result.result_dir)
        if not all((result_dir / name).exists() for name in job.result.files.values()):
            return None  # files were cleaned up; run again
        return job

    def submit(self, request: JobRequest) -> tuple[Job, bool]:
        """Queue a request. Returns ``(job, deduplicated)``.

        An identical request that is queued, running, or finished successfully
        within the dedup window (and whose files still exist) returns the existing
        job instead of running again.
        """
        spec = analyze(
            request,
            image_path=self.resolve_upload(request.image_id),
            input_mesh_path=self.resolve_upload(request.input_mesh_id),
            commercial_use=self.commercial_use,
        )
        fingerprint = spec.fingerprint()
        if existing := self._reusable(fingerprint):
            return existing, True

        job = Job(id=uuid.uuid4().hex[:12], request=request, spec=spec, fingerprint=fingerprint)
        job.log(job.stage, "queued: " + ", ".join(spec.matched_rules))
        self.store.save(job)
        self._active[job.id] = job
        self._queue.put_nowait(job.id)
        return job, False

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.finished:
            return job
        if task := self._running.get(job_id):
            job.cancel_requested = True
            task.cancel()
        else:
            job.status = JobStatus.CANCELLED
            job.error = "cancelled"
            job.finished_at = time.time()
            job.log(job.stage, "cancelled")
            self.store.save(job)
            self._active.pop(job_id, None)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._active.get(job_id) or self.store.get(job_id)

    def list(self, limit: int = 50, status: JobStatus | None = None) -> list[Job]:
        return self.store.list(limit, status)

    def counts(self) -> dict[str, int]:
        return self.store.counts()

    @property
    def queue_size(self) -> int:
        return sum(1 for j in self._active.values() if j.status is JobStatus.QUEUED)

    @property
    def running_count(self) -> int:
        return len(self._running)

    # -- cleanup ---------------------------------------------------------------

    def cleanup(self, now: float | None = None) -> dict[str, int]:
        """Delete finished jobs, their files and unused uploads older than the retention period."""
        if not self.retention_s:
            return {"jobs": 0, "uploads": 0}
        cutoff = (now or time.time()) - self.retention_s
        jobs = 0
        for job in self.store.finished_before(cutoff):
            shutil.rmtree(self.pipeline.jobs_dir / job.id, ignore_errors=True)
            self.store.delete(job.id)
            jobs += 1
        in_use = {Path(p).name for j in self._active.values()
                  for p in (j.spec.image_path, j.spec.input_mesh_path) if p}
        uploads = 0
        for path in self.uploads_dir.glob("*"):
            if path.is_file() and path.name not in in_use and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                uploads += 1
        return {"jobs": jobs, "uploads": uploads}

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                removed = await asyncio.to_thread(self.cleanup)
                if any(removed.values()):
                    log.info("cleanup removed %s", removed)
            except Exception:
                log.exception("cleanup failed")
            await asyncio.sleep(self.cleanup_interval_s)

    # -- internals -------------------------------------------------------------

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._active.get(job_id)
            try:
                if job is None or job.status is not JobStatus.QUEUED:
                    continue  # cancelled while waiting
                task = asyncio.create_task(self.pipeline.run(job))
                self._running[job_id] = task
                try:
                    await task
                except asyncio.CancelledError:
                    # Awaiting a cancelled job also raises here; only re-raise when the
                    # worker itself is being stopped.
                    if asyncio.current_task().cancelling():
                        raise
                    log.info("job %s cancelled", job_id)
                finally:
                    self._running.pop(job_id, None)
                    if job.finished:
                        self._active.pop(job_id, None)
            finally:
                self._queue.task_done()
