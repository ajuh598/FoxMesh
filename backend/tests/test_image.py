import io
import math
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import BLENDER, BLENDER_MODE, fixed_gpu
from foxmesh.adapters.image_relief import (
    bleed_colors, clean_mask, fill_holes, foreground_mask, image_to_mesh, pick_mode,
    remove_diagonal_contacts,
)
from foxmesh.blender.worker import obj_stats
from foxmesh.classifier import analyze
from foxmesh.registry import ModelRegistry, StatsStore
from foxmesh.router import PolicyStore, Router
from foxmesh.schemas import JobRequest, TaskType
from foxmesh.validation import sniff_format


@pytest.fixture
def apple(tmp_path):
    """Red disc with a leaf and a light highlight, on a white JPEG background."""
    img = Image.new("RGB", (320, 240), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((85, 45, 235, 200), fill=(200, 30, 40))
    d.ellipse((150, 20, 190, 55), fill=(40, 160, 60))
    d.ellipse((130, 80, 150, 100), fill=(255, 200, 200))
    path = tmp_path / "apple.jpg"
    img.save(path, quality=90)
    return path


@pytest.fixture
def star(tmp_path):
    """Yellow star on a transparent PNG background."""
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    pts = [(100 + 90 * math.cos(math.pi / 2 + i * math.pi / 5) * (1 if i % 2 == 0 else 0.45),
            100 - 90 * math.sin(math.pi / 2 + i * math.pi / 5) * (1 if i % 2 == 0 else 0.45))
           for i in range(10)]
    ImageDraw.Draw(img).polygon(pts, fill=(250, 200, 30, 255))
    path = tmp_path / "star.png"
    img.save(path)
    return path


def closed(mesh, tmp_path):
    path = tmp_path / "check.obj"
    mesh.write_obj(path)
    return obj_stats(path)


def test_modes_from_prompt():
    assert pick_mode("사과 부조로") == "relief"
    assert pick_mode("로고 키링") == "extrude"
    assert pick_mode("통통한 인형") == "inflate"
    assert pick_mode("사과") is None


def test_foreground_from_background_colour_and_alpha(apple, star):
    small = np.asarray(Image.open(apple).convert("RGBA"))
    mask, method = foreground_mask(small)
    assert method == "background" and 0.2 < mask.mean() < 0.5
    mask, method = foreground_mask(np.asarray(Image.open(star).convert("RGBA")))
    assert method == "alpha"


def test_noisy_image_falls_back_to_full_frame(tmp_path):
    arr = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype("uint8")
    mask, method = foreground_mask(np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)]))
    assert method == "full" and mask.all()


def test_mask_cleanup():
    m = np.zeros((7, 7), bool)
    m[1:6, 1:6] = True
    m[3, 3] = False  # hole
    m[0, 6] = True  # stray speck
    cleaned = clean_mask(m)
    assert cleaned[3, 3] and not cleaned[0, 6]
    assert fill_holes(m)[3, 3]
    diag = np.array([[1, 0], [0, 1]], bool)
    fixed = remove_diagonal_contacts(diag)
    assert fixed.sum() == 3


@pytest.mark.parametrize("prompt,mode", [("", "inflate"), ("부조", "relief"), ("로고", "extrude")])
def test_meshes_are_closed_and_textured(apple, tmp_path, prompt, mode):
    mesh, info = image_to_mesh(apple, prompt)
    assert info.mode == mode and info.mask_method == "background"
    stats = closed(mesh, tmp_path)
    assert stats["faces"] > 1000 and stats["watertight"]
    assert len(mesh.uvs) == len(mesh.vertices)
    assert Image.open(io.BytesIO(mesh.texture_png)).mode == "RGB"


def test_transparent_png_and_size(star, tmp_path):
    mesh, info = image_to_mesh(star, "키링", size_m=0.05)
    assert info.mask_method == "alpha" and info.mode == "extrude"
    xs = [v[0] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    assert max(max(xs) - min(xs), max(zs) - min(zs)) == pytest.approx(0.05, rel=0.05)
    assert closed(mesh, tmp_path)["watertight"]


def test_polygon_budget(apple, tmp_path):
    mesh, _ = image_to_mesh(apple, "", max_polygons=5000)
    assert closed(mesh, tmp_path)["triangles"] <= 5000


def test_texture_edges_take_object_colour(star):
    mesh, _ = image_to_mesh(star, "")
    tex = np.asarray(Image.open(io.BytesIO(mesh.texture_png)))
    # Transparent pixels are black in RGB; after bleeding no black may remain.
    assert tex.sum(axis=-1).min() > 300


def test_bleed_colors_fills_background():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[1:3, 1:3] = (200, 10, 10)
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    out = bleed_colors(rgb, mask)
    assert (out == (200, 10, 10)).all()


def test_glb_output_is_valid(apple, tmp_path):
    mesh, _ = image_to_mesh(apple, "")
    path = tmp_path / "m.glb"
    mesh.write_glb(path)
    assert sniff_format(path).value == "glb"
    import json, struct
    data = path.read_bytes()
    (json_len,) = struct.unpack_from("<I", data, 12)
    gltf = json.loads(data[20:20 + json_len])
    prim = gltf["meshes"][0]["primitives"][0]
    assert {"POSITION", "NORMAL", "TEXCOORD_0"} <= set(prim["attributes"])
    assert gltf["images"][0]["mimeType"] == "image/png"


def test_fast_enough(apple):
    t0 = time.perf_counter()
    image_to_mesh(apple, "")
    assert time.perf_counter() - t0 < 5


def test_router_picks_builtin_for_images(apple):
    reg = ModelRegistry(StatsStore())
    reg.load_builtin()
    spec = analyze(JobRequest(prompt="사진으로 3D 모델"), image_path=str(apple))
    assert spec.task_type is TaskType.IMAGE_TO_3D
    decision = Router(reg, fixed_gpu(24), PolicyStore(), blender_available=False).route(spec)
    assert decision.selected == "image-relief"


def test_api_image_job(tmp_path, apple):
    from fastapi.testclient import TestClient

    from foxmesh.api import create_app
    from foxmesh.config import Settings
    from foxmesh.services import build_services

    services = build_services(Settings(data_dir=tmp_path / "data", blender_executable=None,
                                       models_dir=None), gpu=fixed_gpu(24))
    with TestClient(create_app(services)) as client:
        bad = client.put("/api/uploads", params={"filename": "x.png"}, content=b"not an image")
        assert bad.status_code == 400
        up = client.put("/api/uploads", params={"filename": "apple.jpg"}, content=apple.read_bytes())
        assert up.status_code == 200
        upload_id = up.json()["id"]
        assert client.get(f"/api/uploads/{upload_id}").content == apple.read_bytes()
        job_id = client.post("/api/jobs", json={"image_id": upload_id}).json()["job"]["id"]
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded", job["error"]
        assert job["result"]["model_id"] == "image-relief"
        glb = client.get(f"/api/jobs/{job_id}/files/{job['result']['files']['glb']}").content
        assert glb[:4] == b"glTF"


@pytest.mark.skipif(not BLENDER, reason="FOXMESH_TEST_BLENDER not set")
async def test_image_mesh_through_blender(apple, tmp_path):
    from foxmesh.blender import BlenderWorker
    from foxmesh.validation import validate

    spec = analyze(JobRequest(prompt="게임용"), image_path=str(apple))
    mesh, _ = image_to_mesh(apple, spec.prompt, spec.max_polygons)
    src = tmp_path / "in.glb"
    mesh.write_glb(src)
    worker = BlenderWorker(BLENDER, mode=BLENDER_MODE, thumbnail_size=64, thumbnail_samples=2)
    post = await worker.process(spec, src, tmp_path / "out")
    assert post.ok, post.errors
    assert post.stats["image_textures"] == 1 and post.stats["watertight"]
    assert validate(spec, post).passed
