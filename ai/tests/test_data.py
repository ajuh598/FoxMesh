import json

import numpy as np
import pytest

from conftest import FAST_PRE, sphere_field
from foxmesh_ai.data.dataset import Dataset, split_for
from foxmesh_ai.data.preprocess import load_sample, preprocess_dataset
from foxmesh_ai.data.synthetic import FAMILIES, generate_shape, shape_to_mesh
from foxmesh_ai.geometry import surface_nets, write_glb
from foxmesh_ai.render import render
from foxmesh_ai.text import VOCAB_SIZE, batch_tokenize, tokenize


def test_tokenizer_is_deterministic_and_shares_stems():
    assert tokenize("빨간 의자") == tokenize("빨간  의자")
    a, b = set(tokenize("검")), set(tokenize("검을"))
    assert a & b  # "<검" is shared
    assert all(0 < t < VOCAB_SIZE for t in tokenize("A red CHAIR 의자"))
    rows, length = batch_tokenize(["a", "a much longer prompt"])
    assert len(rows[0]) == len(rows[1]) == length


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_makes_a_closed_coloured_mesh(family):
    shape = generate_shape(np.random.default_rng(0), family)
    mesh = shape_to_mesh(shape, res=40)
    assert mesh.watertight and mesh.volume() > 0
    assert mesh.colors is not None and len(shape.captions) == 6
    assert any("가" <= ch <= "힣" for ch in shape.captions[0])  # Korean caption


def test_render_has_object_and_transparent_background():
    mesh = shape_to_mesh(generate_shape(np.random.default_rng(0), "mug"), res=40)
    img = render(mesh, 30, 20, 64)
    assert img.shape == (64, 64, 4)
    assert 0.1 < (img[..., 3] > 0).mean() < 0.9


def test_dataset_add_dedup_split_validate(tmp_path):
    ds = Dataset.create(tmp_path / "ds", "test", license="CC0-1.0")
    mesh = surface_nets(sphere_field(24))
    item, added = ds.add(mesh, ["ball"], tags=["sphere"], license="CC0-1.0")
    assert added and item.watertight and (ds.root / item.mesh).exists()
    scaled = mesh.copy()
    scaled.vertices = scaled.vertices * 2 + 1  # same asset after normalisation
    again, added = ds.add(scaled, ["a ball", "ball"], tags=["toy"])
    assert not added and again.id == item.id
    assert ds.get(item.id).captions == ["ball", "a ball"] and ds.get(item.id).tags == ["sphere", "toy"]
    reloaded = Dataset(ds.root)
    assert len(reloaded) == 1 and reloaded.fingerprint() == ds.fingerprint()
    assert reloaded.validate(deep=True).ok
    (ds.root / item.mesh).unlink()
    assert not reloaded.validate().ok


def test_split_is_stable_and_proportional():
    ids = [f"{i:016x}" for i in range(2000)]
    splits = [split_for(i, (0.8, 0.1, 0.1)) for i in ids]
    assert abs(splits.count("train") / 2000 - 0.8) < 0.04
    assert splits == [split_for(i, (0.8, 0.1, 0.1)) for i in ids]


def test_import_folder_with_sidecars(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_glb(surface_nets(sphere_field(20)), src / "red_ball.glb")
    write_glb(surface_nets(sphere_field(20, r=0.4, center=(0.2, 0, 0))), src / "cube.glb")
    (src / "cube.json").write_text(json.dumps({"captions": ["작은 상자"], "tags": ["box"], "license": "CC-BY-4.0"}))
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save(src / "red_ball.png")
    (src / "broken.glb").write_bytes(b"x")
    ds = Dataset.create(tmp_path / "ds", "imp")
    counts = ds.import_folder(src, license="CC0-1.0")
    assert counts == {"added": 2, "failed": 1}
    by_caption = {i.captions[0]: i for i in ds}
    assert by_caption["red ball"].images and by_caption["작은 상자"].license == "CC-BY-4.0"


def test_preprocess_outputs(tiny_dataset):
    item = tiny_dataset.items()[0]
    s = load_sample(tiny_dataset.root / "cache" / f"{item.id}.npz")
    assert s["uniform_points"].shape == (FAST_PRE.uniform, 3)
    assert 0 < s["near_occ"].mean() < 1 and s["uniform_occ"].mean() < 0.5
    assert s["renders"].shape == (2, 32, 32, 4) and bool(s["has_colors"])
    assert tiny_dataset.stats()["preprocessed"] == len(tiny_dataset)
    # Already done: nothing to redo.
    assert preprocess_dataset(tiny_dataset, FAST_PRE, workers=1)["processed"] == 0
