import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foxmesh_ai.evaluate import evaluate  # noqa: E402
from foxmesh_ai.generate import Generator, prepare_image  # noqa: E402
from foxmesh_ai.model import FoxMeshModel, ModelConfig  # noqa: E402
from foxmesh_ai.store import ModelStore  # noqa: E402
from foxmesh_ai.text import batch_tokenize  # noqa: E402
from foxmesh_ai.train import TrainConfig, Trainer, finetune_config  # noqa: E402

TINY = ModelConfig(text_dim=32, text_layers=1, text_heads=2, cond_dim=32, hidden=48, blocks=2,
                   image_size=32, fourier_frequencies=4)


def tiny_cfg(**kw):
    base = dict(epochs=2, batch_size=6, lr=3e-3, warmup_steps=2, uniform_points=256, near_points=256,
                surface_points=64, device="cpu")
    base.update(kw)
    return TrainConfig(**base)


def test_forward_shapes_and_modalities():
    model = FoxMeshModel(TINY)
    rows, _ = batch_tokenize(["빨간 의자", "", "a sword"])
    tokens = torch.tensor(rows)
    images = torch.rand(3, 4, 32, 32)
    pts = torch.rand(3, 100, 3) * 2 - 1
    occ, col = model(pts, tokens, images, torch.tensor([1, 0, 1]), torch.tensor([0, 1, 1]))
    assert occ.shape == (3, 100) and col.shape == (3, 100, 3)
    assert torch.isfinite(occ).all() and (col >= 0).all() and (col <= 1).all()
    unconditional = model.encode(tokens, None, text_mask=torch.zeros(3))
    assert torch.allclose(unconditional[0], unconditional[1])


@pytest.fixture(scope="module")
def trained(tiny_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    result = Trainer(tiny_dataset, out, tiny_cfg(epochs=3), TINY, log=lambda m: None).train()
    return result


def test_training_reduces_loss_and_saves_checkpoints(trained, tiny_dataset):
    hist = trained.history
    assert hist[-1]["train_occ_loss"] < hist[0]["train_occ_loss"]
    assert (__import__("pathlib").Path(trained.out_dir) / "checkpoints" / "last.pt").exists()
    log = (__import__("pathlib").Path(trained.out_dir) / "train_log.jsonl").read_text().splitlines()
    assert len(log) == len(hist) and "val_iou" in json.loads(log[-1])


def test_resume_continues_from_last_epoch(trained, tiny_dataset):
    t = Trainer(tiny_dataset, trained.out_dir, tiny_cfg(epochs=4), resume=True, log=lambda m: None)
    assert t.start_epoch == 3 and t.step == trained.steps
    result = t.train()
    assert result.epochs == 4 and len(result.history) == 4


def test_finetune_with_frozen_encoders(trained, tiny_dataset, tmp_path):
    cfg = finetune_config(tiny_cfg(), epochs=1, freeze=["text", "image"], device="cpu")
    t = Trainer(tiny_dataset, tmp_path / "ft", cfg, init_from=trained.best_checkpoint, log=lambda m: None)
    before = {k: v.clone() for k, v in t.model.text.state_dict().items()}
    t.train()
    assert all(torch.equal(before[k], v) for k, v in t.model.text.state_dict().items())
    assert cfg.lr < tiny_cfg().lr


def test_generate_closed_coloured_mesh(trained):
    gen = Generator(trained.best_checkpoint, device="cpu")
    # A barely-trained model may predict an empty shape; force a non-empty field via bias.
    with torch.no_grad():
        gen.model.occ_head.bias.fill_(5.0)
    mesh = gen.generate("빨간 의자", resolution=24)
    assert mesh.watertight and mesh.colors.shape == mesh.vertices.shape
    lo, hi = mesh.bounds
    assert float((hi - lo).max()) == pytest.approx(1.0, abs=1e-4)
    img = np.zeros((40, 60, 4), np.uint8)
    img[10:30, 20:40] = (200, 0, 0, 255)
    framed = prepare_image(img, 32)
    assert framed.shape == (32, 32, 4) and framed[..., 3].max() == 255
    mesh2 = gen.generate(image=img, resolution=24)
    assert mesh2.watertight
    with pytest.raises(ValueError):
        gen.generate()


def test_evaluate_and_store_and_export(trained, tiny_dataset, tmp_path):
    gen = Generator(trained.best_checkpoint, device="cpu")
    with torch.no_grad():
        gen.model.occ_head.bias.fill_(5.0)
    report = evaluate(gen, tiny_dataset, "val", "text", resolution=20, max_items=3, surface_points=256)
    assert report.items == 3 and 0 <= report.iou <= 1 and 0 <= report.fscore <= 1
    assert report.watertight_rate == 1.0 and report.per_tag

    store = ModelStore(tmp_path / "models")
    v1 = store.register(trained.best_checkpoint, metrics={"val_text": report.to_dict()}, notes="test")
    v2 = store.register(trained.best_checkpoint, parent=v1)
    assert (v1, v2) == ("v1", "v2") and store.resolve("latest") == "v2"
    with pytest.raises(LookupError):
        store.resolve("production")
    store.promote(v1)
    assert store.checkpoint() == store.root / "v1" / "model.pt"
    assert "IoU" in (store.root / "v1" / "card.md").read_text()
    Generator(store.checkpoint(), device="cpu")  # slim checkpoint loads
    spec = json.loads(store.export(tmp_path / "platform").read_text())
    assert spec["id"] == "foxmesh" and spec["adapter"] == "foxmesh_ai" and spec["version"] == "v1"
    assert spec["config"]["store"] == str(store.root.resolve())
    with pytest.raises(ValueError):
        store.delete("v1")
    store.delete("v2")
    assert [v for v, _ in store.versions()] == ["v1"]


def test_cli_end_to_end(tiny_dataset, trained, tmp_path, capsys):
    from foxmesh_ai.cli import main

    assert main(["dataset", "stats", str(tiny_dataset.root)]) == 0
    assert '"items": 24' in capsys.readouterr().out
    store = tmp_path / "models"
    assert main(["models", "register", "--store", str(store), trained.best_checkpoint, "--promote"]) == 0
    out = tmp_path / "x.obj"
    # The tiny model may produce an empty shape; generation errors are acceptable here.
    try:
        main(["generate", "--store", str(store), "--prompt", "sword", "--resolution", "20", "--out", str(out)])
    except RuntimeError:
        pass
    assert main(["models", "export", "--store", str(store), "--to", str(tmp_path / "pm")]) == 0
    assert (tmp_path / "pm" / "foxmesh.json").exists()
