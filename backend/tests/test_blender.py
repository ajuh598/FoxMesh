"""Real Blender runs. Opt in with FOXMESH_TEST_BLENDER=/path/to/blender (or a Python with bpy
plus FOXMESH_TEST_BLENDER_MODE=bpy)."""

import pytest

from conftest import BLENDER, BLENDER_MODE
from foxmesh.adapters.procedural import Mesh, box, generate_mesh, sphere
from foxmesh.blender import BlenderWorker
from foxmesh.classifier import analyze
from foxmesh.schemas import JobRequest, OutputFormat
from foxmesh.validation import sniff_format, validate

pytestmark = pytest.mark.skipif(not BLENDER, reason="FOXMESH_TEST_BLENDER not set")


def worker(**kw):
    return BlenderWorker(BLENDER, mode=BLENDER_MODE, thumbnail_size=64, thumbnail_samples=2, **kw)


async def test_game_asset_pipeline(tmp_path):
    spec = analyze(JobRequest(prompt="게임용 중세풍 검"))
    src = tmp_path / "sword.glb"
    generate_mesh(spec.prompt)[1].write_glb(src)
    post = await worker().process(spec, src, tmp_path / "out")
    assert post.ok, post.errors
    assert set(post.outputs) == {OutputFormat.GLB, OutputFormat.FBX}
    for fmt, path in post.outputs.items():
        assert sniff_format(path) is fmt
    assert post.stats["has_uv"] and post.stats["materials"] == 1
    assert post.stats["bbox_min"][2] == pytest.approx(0, abs=1e-6)  # standing on the ground
    assert post.thumbnail and post.thumbnail.stat().st_size > 0
    assert validate(spec, post).passed


async def test_decimate_to_budget(tmp_path):
    spec = analyze(JobRequest(prompt="sphere 300 polygons", output_formats=[OutputFormat.GLB]))
    src = tmp_path / "ball.obj"
    Mesh().add(sphere(0.5, 32, 64), "ball").write_obj(src)
    post = await worker(thumbnails=False).process(spec, src, tmp_path / "out")
    info = post.report["steps"]["decimate"]["info"]
    assert info["applied"] and info["triangles_before"] > 3000
    assert post.stats["triangles"] <= 300 * 1.05
    assert validate(spec, post).passed


async def test_print_pipeline_repairs_and_scales(tmp_path):
    spec = analyze(JobRequest(prompt="3D 프린팅용 5cm 상자"))
    # An open box (top face removed) must be closed for printing.
    mesh = box((1, 1, 1))
    mesh.faces.pop(1)
    src = tmp_path / "open.obj"
    Mesh().add(mesh, "open").write_obj(src)
    post = await worker(thumbnails=False).process(spec, src, tmp_path / "out")
    check = post.report["steps"]["check_manifold"]["info"]
    assert check["non_manifold_before"] > 0 and check["watertight"]
    assert max(post.stats["dimensions_m"]) == pytest.approx(0.05, rel=1e-3)
    assert sniff_format(post.outputs[OutputFormat.STL]) is OutputFormat.STL
    assert validate(spec, post).passed


async def test_bad_input_reports_failure(tmp_path):
    spec = analyze(JobRequest(prompt="cube"))
    src = tmp_path / "broken.glb"
    src.write_bytes(b"not a gltf file")
    post = await worker().process(spec, src, tmp_path / "out")
    assert not post.ok and any("import" in e for e in post.errors)
    assert not validate(spec, post).passed
