"""Platform integration of FoxMesh's own model (skipped without foxmesh-ai and torch)."""

import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("foxmesh_ai.generate")

from fastapi.testclient import TestClient  # noqa: E402

from conftest import fixed_gpu  # noqa: E402
from foxmesh.api import create_app  # noqa: E402
from foxmesh.config import Settings  # noqa: E402
from foxmesh.services import build_services  # noqa: E402


@pytest.fixture
def model_dir(tmp_path):
    from foxmesh_ai.model import FoxMeshModel, ModelConfig
    from foxmesh_ai.store import ModelStore

    cfg = ModelConfig(text_dim=32, text_layers=1, text_heads=2, cond_dim=32, hidden=32, blocks=1,
                      image_size=32, fourier_frequencies=2)
    model = FoxMeshModel(cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"ema": model.state_dict(), "model_config": cfg.to_dict(), "step": 0,
                "dataset": {"name": "test", "fingerprint": "x"}}, ckpt)
    store = ModelStore(tmp_path / "store")
    store.promote(store.register(ckpt, metrics={"val_text": {
        "split": "val", "modality": "text", "items": 1, "iou": 0.9, "chamfer": 0.01, "fscore": 0.95,
        "watertight_rate": 1.0, "mean_generation_s": 0.2}}))
    models = tmp_path / "platform_models"
    store.export(models, resolution=24)
    return models, store


def test_registered_model_is_selected_and_runs(tmp_path, model_dir, monkeypatch):
    models, store = model_dir
    settings = Settings(data_dir=tmp_path / "data", blender_executable=None, models_dir=models,
                        database_url="memory://")
    services = build_services(settings, gpu=fixed_gpu(0))
    view = services.registry.view("foxmesh")
    assert view.available, view.unavailable_reason
    assert view.spec.version == "v1" and view.spec.adapter == "foxmesh_ai"

    # An untrained model may predict an empty shape; substitute a sphere for the field.
    from foxmesh_ai.generate import Generator
    original = Generator.field

    def sphere_field(self, cond, resolution=64, chunk=65536):
        import numpy as np
        c = (np.arange(resolution) + 0.5) / resolution * 2 - 1
        x, y, z = np.meshgrid(c, c, c, indexing="ij")
        return 0.6 - np.sqrt(x ** 2 + y ** 2 + z ** 2)
    monkeypatch.setattr(Generator, "field", sphere_field)

    with TestClient(create_app(services)) as client:
        body = client.post("/api/analyze", json={"prompt": "빨간 공"}).json()
        assert body["routing"]["selected"] == "foxmesh", body["routing"]
        job_id = client.post("/api/jobs", json={"prompt": "빨간 공"}).json()["job"]["id"]
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded", job["error"]
        assert job["result"]["model_id"] == "foxmesh"
        glb = client.get(f"/api/jobs/{job_id}/files/{job['result']['files']['glb']}").content
        assert glb[:4] == b"glTF"
    monkeypatch.setattr(Generator, "field", original)


def test_missing_checkpoint_makes_model_unavailable(tmp_path, model_dir):
    models, store = model_dir
    (store.root / "v1" / "model.pt").unlink()
    services = build_services(Settings(data_dir=tmp_path / "d", blender_executable=None,
                                       models_dir=models, database_url="memory://"), gpu=fixed_gpu(0))
    assert "checkpoint missing" in services.registry.view("foxmesh").unavailable_reason
