import time

import pytest
from fastapi.testclient import TestClient

from conftest import fixed_gpu
from foxmesh.api import create_app
from foxmesh.benchmark import BenchmarkResult, BenchmarkRun, summarize
from foxmesh.config import Settings
from foxmesh.services import build_services


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, blender_executable=None, models_dir=None,
                        database_url="memory://")
    with TestClient(create_app(build_services(settings, gpu=fixed_gpu(24)))) as c:
        yield c


def wait_benchmark(client, run_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/benchmarks/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("benchmark did not finish")


def test_summary_math():
    run = BenchmarkRun(id="x", model_ids=["a"], case_ids=["c1", "c2"], results=[
        BenchmarkResult(model_id="a", case_id="c1", ok=True, generation_s=1.0, validation_score=1.0),
        BenchmarkResult(model_id="a", case_id="c2", ok=True, generation_s=3.0, validation_score=0.5),
        BenchmarkResult(model_id="a", case_id="c2", ok=False, error="boom"),
    ])
    (s,) = summarize(run)
    assert s.runs == 3 and s.successes == 2 and s.success_rate == pytest.approx(0.667, abs=1e-3)
    assert s.mean_generation_s == 2.0 and s.p90_generation_s == 3.0
    assert s.mean_validation_score == 0.75


def test_benchmark_runs_and_feeds_router_stats(client):
    assert {c["id"] for c in client.get("/api/benchmarks/cases").json()} >= {"image-apple"}
    # Without Blender only GLB-only cases can run; pick those explicitly.
    resp = client.post("/api/benchmarks", json={"case_ids": ["text-prototype-chair", "image-apple"],
                                                "repeats": 2})
    assert resp.status_code == 202
    run = wait_benchmark(client, resp.json()["id"])
    assert run["status"] == "done", run["error"]
    summary = {s["model_id"]: s for s in run["summary"]}
    assert summary["procedural-primitives"]["runs"] == 2
    assert summary["procedural-primitives"]["skipped_cases"] == ["image-apple"]
    assert summary["image-relief"]["success_rate"] == 1.0
    assert summary["image-relief"]["mean_validation_score"] is not None
    stats = client.get("/api/models/image-relief").json()["stats"]
    assert stats["runs"] == 2 and stats["mean_runtime_s"] is not None
    assert client.get("/api/benchmarks").json()[0]["id"] == run["id"]
    # Benchmark jobs do not show up as user jobs.
    assert client.get("/api/jobs").json() == []


def test_benchmark_validation(client):
    assert client.post("/api/benchmarks", json={"model_ids": ["nope"]}).status_code == 400
    assert client.post("/api/benchmarks", json={"case_ids": ["nope"]}).status_code == 400
    assert client.get("/api/benchmarks/missing").status_code == 404


def test_metrics_endpoint(client):
    text = client.get("/api/metrics").text
    assert '# TYPE foxmesh_jobs gauge' in text
    assert 'foxmesh_model_available{model="procedural-primitives"} 1' in text
    assert 'foxmesh_gpu_vram_total_gb 24.0' in text


def test_result_files_are_cacheable(client):
    job_id = client.post("/api/jobs", json={"prompt": "quick cube prototype"}).json()["job"]["id"]
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "queued" and job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job["error"]
    resp = client.get(f"/api/jobs/{job_id}/files/{job['result']['files']['glb']}")
    assert "immutable" in resp.headers["cache-control"]
    assert resp.headers.get("etag")
