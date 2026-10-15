from pathlib import Path

from foxmesh.adapters.procedural import generate_mesh
from foxmesh.blender import LitePostProcessor, Step, plan_steps
from foxmesh.blender.worker import PostProcessResult, obj_stats
from foxmesh.classifier import analyze
from foxmesh.schemas import JobRequest, OutputFormat
from foxmesh.validation import sniff_format, validate


def steps(prompt, **kw):
    return plan_steps(analyze(JobRequest(prompt=prompt), **kw))


def test_steps_follow_profile():
    game = steps("게임용 검")
    assert {Step.UV, Step.MATERIALS, Step.DECIMATE, Step.EXPORT} <= set(game)
    assert Step.CHECK_MANIFOLD not in game

    printing = steps("3D 프린팅용 컵")
    assert Step.CHECK_MANIFOLD in printing and Step.MATERIALS not in printing

    proto = steps("quick prototype chair")
    assert Step.DECIMATE not in proto and Step.UV not in proto

    assert Step.SAVE_BLEND in steps("고품질 투구")


def test_steps_for_existing_mesh():
    preview = steps("썸네일 렌더", input_mesh_path="/x.glb")
    assert preview[:2] == [Step.IMPORT, Step.INSPECT] and Step.THUMBNAIL in preview
    convert = steps("FBX로 변환", input_mesh_path="/x.glb")
    assert Step.THUMBNAIL not in convert and Step.NORMALIZE not in convert


def test_order_is_stable():
    order = list(Step)
    s = steps("3D 프린팅용 10cm 컵 1000 폴리곤, 블렌더 파일")
    assert s == sorted(s, key=order.index)


def test_procedural_meshes_are_closed(tmp_path):
    for prompt in ["sword", "table", "tree", "cone", "sphere", "house", "unknown thing"]:
        _, mesh = generate_mesh(prompt)
        path = tmp_path / "m.obj"
        mesh.write_obj(path)
        stats = obj_stats(path)
        assert stats["faces"] > 0
        assert stats["watertight"], prompt


def test_procedural_respects_polygon_budget():
    _, small = generate_mesh("sphere", max_polygons=300)
    _, large = generate_mesh("sphere")
    tris = lambda m: sum(len(f) - 2 for f in m.faces)
    assert tris(small) <= 300 < tris(large)


def test_sniff_formats(tmp_path):
    cases = {
        "a.glb": b"glTF\x02\x00\x00\x00" + b"\x00" * 20,
        "a.fbx": b"Kaydara FBX Binary  \x00" + b"\x00" * 20,
        "a.ply": b"ply\nformat ascii 1.0\n",
        "a.stl": b"\x00" * 80 + (1).to_bytes(4, "little") + b"\x00" * 50,
        "a.obj": b"# c\nv 0 0 0\nv 1 0 0\n",
    }
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        assert sniff_format(tmp_path / name).value == name.split(".")[1]
    (tmp_path / "fake.glb").write_bytes(b"not a model")
    assert sniff_format(tmp_path / "fake.glb") is None


async def test_lite_postprocess_and_validation(tmp_path):
    _, mesh = generate_mesh("cube")
    src = tmp_path / "in.obj"
    mesh.write_obj(src)
    spec = analyze(JobRequest(prompt="cube", output_formats=[OutputFormat.OBJ]))
    spec.output_formats = [OutputFormat.OBJ]
    post = await LitePostProcessor().process(spec, src, tmp_path / "out")
    result = validate(spec, post)
    assert post.ok and result.passed, result.checks

    spec.output_formats = [OutputFormat.GLB]
    post = await LitePostProcessor().process(spec, src, tmp_path / "out2")
    assert not post.ok and not validate(spec, post).passed


def test_validation_polygon_budget_and_watertight(tmp_path):
    spec = analyze(JobRequest(prompt="3D 프린팅 컵 100 폴리곤", output_formats=[OutputFormat.STL]))
    out = tmp_path / "model.stl"
    out.write_bytes(b"\x00" * 80 + (1).to_bytes(4, "little") + b"\x00" * 50)
    post = PostProcessResult(
        ok=True, engine="blender", steps=[], outputs={OutputFormat.STL: out, OutputFormat.GLB: out},
        report={"stats": {"faces": 500, "triangles": 900, "watertight": False,
                          "non_manifold_edges": 4}},
    )
    result = validate(spec, post)
    failed = {c.name for c in result.failures}
    assert {"mesh:polygon_budget", "mesh:watertight", "output:glb"} <= failed
    assert not result.passed and 0 < result.score < 1
