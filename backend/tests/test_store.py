import asyncio
import os
import time
import uuid

import pytest

from conftest import fixed_gpu, make_model
from foxmesh.adapters import AdapterRegistry, GenerationAdapter, GenerationResult
from foxmesh.adapters.procedural import generate_mesh
from foxmesh.blender import LitePostProcessor
from foxmesh.jobs import JobManager
from foxmesh.pipeline import Job, JobStatus, Pipeline
from foxmesh.router import PolicyStore, Router
from foxmesh.schemas import JobRequest, OutputFormat, Profile
from foxmesh.store import MemoryJobStore, open_store, sqlite_store

PG_URL = os.environ.get("FOXMESH_TEST_DATABASE_URL")


def make_job(prompt="cube", status=JobStatus.QUEUED, created=None, finished=None) -> Job:
    from foxmesh.classifier import analyze

    spec = analyze(JobRequest(prompt=prompt))
    return Job(id=uuid.uuid4().hex[:12], request=JobRequest(prompt=prompt), spec=spec,
               fingerprint=spec.fingerprint(), status=status,
               created_at=created or time.time(), finished_at=finished)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryJobStore()
    elif request.param == "sqlite":
        s = sqlite_store(tmp_path / "jobs.db")
        yield s
        s.close()
    else:
        if not PG_URL:
            pytest.skip("FOXMESH_TEST_DATABASE_URL not set")
        s = open_store(PG_URL, tmp_path)
        s._execute("DELETE FROM jobs")
        yield s
        s.close()


def test_store_roundtrip_and_queries(store):
    old = make_job("old", JobStatus.SUCCEEDED, created=100, finished=110)
    new = make_job("new", created=200)
    store.save(old)
    store.save(new)
    assert store.get(old.id).spec.prompt == "old"
    assert [j.id for j in store.list(10)] == [new.id, old.id]
    assert [j.id for j in store.list(10, JobStatus.SUCCEEDED)] == [old.id]
    assert [j.id for j in store.unfinished()] == [new.id]
    assert store.counts()["queued"] == 1 and store.counts()["succeeded"] == 1
    assert [j.id for j in store.finished_before(150)] == [old.id]

    new.status = JobStatus.RUNNING
    new.log(new.stage, "progress")
    store.save(new)  # upsert
    assert store.get(new.id).status is JobStatus.RUNNING
    assert len(store.list(10)) == 2

    twin = make_job("old", created=300)
    store.save(twin)
    assert store.latest_with_fingerprint(old.fingerprint).id == twin.id
    store.delete(old.id)
    assert store.get(old.id) is None


def test_sqlite_survives_reopen(tmp_path):
    path = tmp_path / "jobs.db"
    job = make_job()
    s = sqlite_store(path)
    s.save(job)
    s.close()
    assert sqlite_store(path).get(job.id).id == job.id


def test_open_store_urls(tmp_path):
    assert isinstance(open_store("memory://", tmp_path), MemoryJobStore)
    assert (tmp_path / "foxmesh.db").exists() is False
    open_store(None, tmp_path)
    assert (tmp_path / "foxmesh.db").exists()
    with pytest.raises(ValueError):
        open_store("mysql://x", tmp_path)


class GatedAdapter(GenerationAdapter):
    """Blocks until released, so tests can stop the manager mid-job."""

    key = "fake"

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def generate(self, task, model, workdir):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        path = workdir / "out.glb"
        generate_mesh("cube")[1].write_glb(path)
        return GenerationResult(path=path, format=OutputFormat.GLB)


def manager(registry, tmp_path, store, adapter, **kw):
    registry.register(make_model("m", outputs=["glb"], runtime="cpu"))
    router = Router(registry, fixed_gpu(24), PolicyStore(), blender_available=False)
    pipeline = Pipeline(registry, AdapterRegistry([adapter], load_plugins=False), router,
                        fixed_gpu(24), LitePostProcessor(), tmp_path / "jobs")
    return JobManager(pipeline, tmp_path / "uploads", store=store, concurrency=1, **kw)


def glb_request(prompt="cube"):
    return JobRequest(prompt=prompt, profile=Profile.PROTOTYPE)  # GLB only


async def test_shutdown_requeues_and_restart_resumes(registry, tmp_path):
    (tmp_path / "uploads").mkdir()
    store = sqlite_store(tmp_path / "jobs.db")
    adapter = GatedAdapter()
    m1 = manager(registry, tmp_path, store, adapter)
    await m1.start()
    job, _ = m1.submit(glb_request())
    await asyncio.wait_for(adapter.started.wait(), 5)
    await m1.stop()  # graceful shutdown mid-generation
    saved = store.get(job.id)
    assert saved.status is JobStatus.QUEUED and "shutdown" in saved.events[-1].message

    adapter2 = GatedAdapter()
    adapter2.release.set()
    m2 = manager(registry, tmp_path, store, adapter2)
    await m2.start()
    done = await m2.wait(job.id, 10)
    await m2.stop()
    assert done.status is JobStatus.SUCCEEDED, done.error
    assert done.recoveries == 0 and len(done.attempts) == 2  # interrupted one + the resumed one


async def test_crash_recovery_is_bounded(registry, tmp_path):
    (tmp_path / "uploads").mkdir()
    store = MemoryJobStore()
    crashed = make_job(status=JobStatus.RUNNING)
    crashed.recoveries = 1  # already re-run once after a crash
    store.save(crashed)
    adapter = GatedAdapter()
    m = manager(registry, tmp_path, store, adapter)
    assert m.recover() == 0
    assert store.get(crashed.id).status is JobStatus.FAILED


async def test_dedup_survives_restart_until_files_are_gone(registry, tmp_path):
    (tmp_path / "uploads").mkdir()
    store = sqlite_store(tmp_path / "jobs.db")
    adapter = GatedAdapter()
    adapter.release.set()
    m1 = manager(registry, tmp_path, store, adapter)
    await m1.start()
    first, _ = m1.submit(glb_request())
    done = await m1.wait(first.id, 10)
    await m1.stop()
    assert done.status is JobStatus.SUCCEEDED, done.error

    m2 = manager(registry, tmp_path, store, adapter)
    again, dedup = m2.submit(glb_request())
    assert dedup and again.id == first.id

    import shutil
    shutil.rmtree(done.result.result_dir)
    third, dedup = m2.submit(glb_request())
    assert not dedup and third.id != first.id


async def test_cancel_queued_and_running(registry, tmp_path):
    (tmp_path / "uploads").mkdir()
    store = MemoryJobStore()
    adapter = GatedAdapter()
    m = manager(registry, tmp_path, store, adapter)
    await m.start()
    running, _ = m.submit(glb_request("cube one"))
    queued, _ = m.submit(glb_request("cube two"))
    await asyncio.wait_for(adapter.started.wait(), 5)
    assert m.cancel(queued.id).status is JobStatus.CANCELLED
    m.cancel(running.id)
    done = await m.wait(running.id, 5)
    await m.stop()
    assert done.status is JobStatus.CANCELLED
    assert store.get(queued.id).status is JobStatus.CANCELLED
    assert adapter.calls == 1


def test_retention_cleanup(registry, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    store = MemoryJobStore()
    m = manager(registry, tmp_path, store, GatedAdapter(), retention_s=86400)
    now = time.time()
    old = make_job("old", JobStatus.SUCCEEDED, created=now - 3 * 86400, finished=now - 2 * 86400)
    recent = make_job("recent", JobStatus.SUCCEEDED, created=now - 100, finished=now - 50)
    store.save(old)
    store.save(recent)
    (tmp_path / "jobs" / old.id).mkdir(parents=True)
    stale = uploads / "abc.png"
    stale.write_bytes(b"x")
    os.utime(stale, (now - 3 * 86400, now - 3 * 86400))
    fresh = uploads / "def.png"
    fresh.write_bytes(b"x")
    assert m.cleanup(now) == {"jobs": 1, "uploads": 1}
    assert store.get(old.id) is None and store.get(recent.id) is not None
    assert not (tmp_path / "jobs" / old.id).exists()
    assert not stale.exists() and fresh.exists()
