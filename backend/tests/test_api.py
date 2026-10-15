import time

import pytest
from fastapi.testclient import TestClient

from conftest import fixed_gpu
from foxmesh.api import create_app
from foxmesh.config import Settings
from foxmesh.services import build_services


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, blender_executable=None, models_dir=None)
    services = build_services(settings, gpu=fixed_gpu(24))
    with TestClient(create_app(services)) as c:
        yield c


def wait(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_and_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    system = client.get("/api/system").json()
    assert system["blender"]["available"] is False
    assert system["gpu"]["total_vram_gb"] == 24


def test_analyze_explains_selection(client):
    body = client.post("/api/analyze", json={"prompt": "빠르게 의자 시제품"}).json()
    assert body["spec"]["profile"] == "prototype"
    assert body["routing"]["selected"] == "procedural-primitives"
    rejected = {r["model_id"]: r["reasons"] for r in body["routing"]["rejected"]}
    assert "unavailable: disabled" in rejected["trellis"]


def test_analyze_reports_no_candidate(client):
    # Game assets need FBX, which the procedural model cannot write without Blender.
    body = client.post("/api/analyze", json={"prompt": "게임용 검"}).json()
    assert body["routing"]["selected"] is None and "no model" in body["error"]


def test_job_lifecycle_and_download(client):
    resp = client.post("/api/jobs", json={"prompt": "a simple wooden table"})
    assert resp.status_code == 202
    job = wait(client, resp.json()["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    assert set(job["result"]["files"]) == {"glb"}
    name = job["result"]["files"]["glb"]
    data = client.get(f"/api/jobs/{job['id']}/files/{name}").content
    assert data[:4] == b"glTF"
    assert client.get(f"/api/jobs/{job['id']}/files/job.json").status_code == 404

    jobs = client.get("/api/jobs").json()
    assert jobs[0]["id"] == job["id"] and jobs[0]["model_id"] == "procedural-primitives"
    dash = client.get("/api/dashboard").json()
    assert dash["jobs"]["succeeded"] == 1
    model = client.get("/api/models/procedural-primitives").json()
    assert model["stats"]["runs"] == 1


def test_duplicate_requests_share_a_job(client):
    first = client.post("/api/jobs", json={"prompt": "Red  cube"}).json()
    second = client.post("/api/jobs", json={"prompt": "red cube"}).json()
    assert second["deduplicated"] and second["job"]["id"] == first["job"]["id"]


def test_submit_validation(client):
    assert client.post("/api/jobs", json={"prompt": "  "}).status_code == 400
    assert client.post("/api/jobs", json={"prompt": "x", "image_id": "../etc"}).status_code == 400


def test_upload_and_convert_without_blender(client, tmp_path):
    from foxmesh.adapters.procedural import generate_mesh

    path = tmp_path / "in.obj"
    generate_mesh("cube")[1].write_obj(path)
    up = client.put("/api/uploads", params={"filename": "cube.obj"}, content=path.read_bytes())
    assert up.status_code == 200
    resp = client.post("/api/jobs", json={"prompt": "OBJ로 변환", "input_mesh_id": up.json()["id"]})
    job = wait(client, resp.json()["job"]["id"])
    assert job["spec"]["task_type"] == "convert"
    assert job["status"] == "succeeded", job["error"]
    assert client.put("/api/uploads", params={"filename": "x.exe"}, content=b"1").status_code == 400


def test_model_management(client):
    resp = client.patch("/api/models/trellis",
                        json={"enabled": True, "config": {"endpoint": "http://gpu:9000"}})
    assert resp.status_code == 200 and resp.json()["available"] is True
    assert client.patch("/api/models/nope", json={"enabled": True}).status_code == 404
    new = {"id": "mine", "name": "Mine", "version": "0.1", "adapter": "http",
           "tasks": ["generate_3d"], "inputs": ["text"], "outputs": ["glb"]}
    assert client.post("/api/models", json=new).status_code == 201
    assert client.post("/api/models", json=new).status_code == 409
    view = client.get("/api/models/mine").json()
    assert view["unavailable_reason"].startswith("no inference endpoint")


def test_policies(client):
    policies = client.get("/api/policies").json()
    assert set(policies) == {"prototype", "balanced", "high_quality", "game_asset", "print_3d"}
    custom = {"weights": {"accuracy": 0, "speed": 1, "quality": 0, "stability": 0, "resource": 0},
              "min_quality": 0.1, "reference_runtime_s": 10}
    assert client.put("/api/policies/balanced", json=custom).json()["min_quality"] == 0.1
    assert client.delete("/api/policies/balanced").json()["min_quality"] == 0.3
