"""Durable job storage.

Jobs are stored as JSON documents with a few indexed columns (status,
fingerprint, timestamps) so they survive restarts, duplicate detection works
across restarts, and interrupted jobs can be recovered.

``FOXMESH_DATABASE_URL`` selects the backend:

* ``sqlite:///path/to/foxmesh.db`` (default: ``<data_dir>/foxmesh.db``)
* ``postgresql://user:pass@host/db`` (needs ``psycopg``)
* ``memory://`` (tests, throwaway runs)
"""

from __future__ import annotations

import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .pipeline import Job, JobStatus

FINISHED = (JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value)


class JobStore(ABC):
    @abstractmethod
    def save(self, job: Job) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> Job | None: ...

    @abstractmethod
    def list(self, limit: int = 50, status: JobStatus | None = None) -> list[Job]: ...

    @abstractmethod
    def latest_with_fingerprint(self, fingerprint: str) -> Job | None: ...

    @abstractmethod
    def unfinished(self) -> list[Job]: ...

    @abstractmethod
    def counts(self) -> dict[str, int]: ...

    @abstractmethod
    def finished_before(self, ts: float) -> list[Job]: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...

    def close(self) -> None:
        pass


class MemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def save(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50, status: JobStatus | None = None) -> list[Job]:
        jobs = [j for j in self._jobs.values() if status is None or j.status is status]
        return sorted(jobs, key=lambda j: -j.created_at)[:limit]

    def latest_with_fingerprint(self, fingerprint: str) -> Job | None:
        matches = [j for j in self._jobs.values() if j.fingerprint == fingerprint]
        return max(matches, key=lambda j: j.created_at, default=None)

    def unfinished(self) -> list[Job]:
        return sorted((j for j in self._jobs.values() if not j.finished), key=lambda j: j.created_at)

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in JobStatus}
        for job in self._jobs.values():
            out[job.status.value] += 1
        return out

    def finished_before(self, ts: float) -> list[Job]:
        return [j for j in self._jobs.values() if j.finished and (j.finished_at or 0) < ts]

    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


class SqlJobStore(JobStore):
    """One table, JSON payload plus indexed columns. Works on SQLite and PostgreSQL."""

    SCHEMA = [
        """CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            finished_at DOUBLE PRECISION,
            data TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS jobs_fingerprint ON jobs (fingerprint, created_at)",
        "CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created_at)",
        "CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created_at)",
    ]

    def __init__(self, conn, placeholder: str):
        self._p = placeholder
        self._lock = threading.Lock()
        self._conn = conn
        for statement in self.SCHEMA:
            self._execute(statement)

    def _sql(self, statement: str) -> str:
        return statement.replace("?", self._p)

    def _run(self, statement: str, params: tuple, fetch: bool) -> list[tuple]:
        with self._lock:
            try:
                cur = self._conn.execute(self._sql(statement), params)
                rows = cur.fetchall() if fetch else []
                self._conn.commit()
                return rows
            except Exception:
                # PostgreSQL refuses further statements until a failed transaction is rolled back.
                self._conn.rollback()
                raise

    def _execute(self, statement: str, params: tuple = ()) -> None:
        self._run(statement, params, fetch=False)

    def _query(self, statement: str, params: tuple = ()) -> list[tuple]:
        return self._run(statement, params, fetch=True)

    def _jobs(self, statement: str, params: tuple = ()) -> list[Job]:
        return [Job.model_validate_json(row[0]) for row in self._query(statement, params)]

    def save(self, job: Job) -> None:
        self._execute(
            "INSERT INTO jobs (id, fingerprint, status, created_at, finished_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET "
            "status = excluded.status, finished_at = excluded.finished_at, data = excluded.data",
            (job.id, job.fingerprint, job.status.value, job.created_at, job.finished_at,
             job.model_dump_json()),
        )

    def get(self, job_id: str) -> Job | None:
        jobs = self._jobs("SELECT data FROM jobs WHERE id = ?", (job_id,))
        return jobs[0] if jobs else None

    def list(self, limit: int = 50, status: JobStatus | None = None) -> list[Job]:
        if status is None:
            return self._jobs("SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return self._jobs("SELECT data FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                          (status.value, limit))

    def latest_with_fingerprint(self, fingerprint: str) -> Job | None:
        jobs = self._jobs("SELECT data FROM jobs WHERE fingerprint = ? "
                          "ORDER BY created_at DESC LIMIT 1", (fingerprint,))
        return jobs[0] if jobs else None

    def unfinished(self) -> list[Job]:
        return self._jobs("SELECT data FROM jobs WHERE status NOT IN (?, ?, ?) ORDER BY created_at",
                          FINISHED)

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in JobStatus}
        for status, count in self._query("SELECT status, COUNT(*) FROM jobs GROUP BY status"):
            out[status] = int(count)
        return out

    def finished_before(self, ts: float) -> list[Job]:
        return self._jobs("SELECT data FROM jobs WHERE status IN (?, ?, ?) AND finished_at < ?",
                          (*FINISHED, ts))

    def delete(self, job_id: str) -> None:
        self._execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def sqlite_store(path: Path) -> SqlJobStore:
    path.parent.mkdir(parents=True, exist_ok=True)

    # The API runs handlers in worker threads; access is serialised by the store's lock.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return SqlJobStore(conn, "?")


def postgres_store(url: str) -> SqlJobStore:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("PostgreSQL storage needs psycopg: pip install 'psycopg[binary]'") from exc
    return SqlJobStore(psycopg.connect(url), "%s")


def open_store(url: str | None, data_dir: Path) -> JobStore:
    if not url:
        return sqlite_store(data_dir / "foxmesh.db")
    if url.startswith("memory:"):
        return MemoryJobStore()
    if url.startswith("sqlite:///"):
        return sqlite_store(Path(url[len("sqlite:///"):]))
    if url.startswith(("postgresql://", "postgres://")):
        return postgres_store(url)
    raise ValueError(f"unsupported FOXMESH_DATABASE_URL: {url}")
