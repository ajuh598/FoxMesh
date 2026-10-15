import numpy as np
import pytest

from conftest import sphere_field
from foxmesh_ai.geometry import (
    TriMesh, chamfer_and_fscore, grid_lookup, load_mesh, make_well_composed, normalize,
    sample_surface, surface_nets, taubin_smooth, voxelize, write_glb, write_obj,
)


def test_surface_nets_sphere_is_closed_outward_and_accurate():
    m = surface_nets(sphere_field(48))
    assert m.watertight
    assert m.volume() == pytest.approx(4 / 3 * np.pi * 0.6 ** 3, rel=0.02)


def test_surface_nets_is_manifold_on_noise():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = surface_nets(rng.normal(size=(16, 16, 16)))
        assert m.edge_manifold_stats() == (0, 0)
        assert m.volume() > 0


def test_well_composed_removes_diagonal_contacts():
    g = np.zeros((4, 4, 4), bool)
    g[1, 1, 1] = g[2, 2, 2] = True  # touch only at a corner
    fixed = make_well_composed(g)
    assert fixed.sum() > 2 and fixed[1, 1, 1] and fixed[2, 2, 2]


def test_voxelize_matches_analytic_shape():
    m = surface_nets(sphere_field(48, r=0.5, center=(0.2, -0.1, 0.0)))
    g = voxelize(m, 32)
    assert g.mean() == pytest.approx(4 / 3 * np.pi * 0.125 / 8, rel=0.08)
    pts = np.array([[0.2, -0.1, 0.0], [-0.8, 0.8, 0.8]])
    assert grid_lookup(g, pts).tolist() == [True, False]


def test_torus_hole_is_empty():
    c = (np.arange(40) + 0.5) / 40 * 2 - 1
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    m = surface_nets(0.2 - np.sqrt((np.sqrt(x ** 2 + z ** 2) - 0.5) ** 2 + y ** 2))
    g = voxelize(m, 32)
    assert not g[16, 16, 16] and g.mean() > 0.03


def test_normalize_and_sampling():
    m = surface_nets(sphere_field(24))
    m.vertices = m.vertices * 3 + 5
    n, info = normalize(m)
    lo, hi = n.bounds
    assert np.allclose((hi - lo).max(), 1.8, atol=1e-4) and np.allclose((lo + hi) / 2, 0, atol=1e-4)
    m.colors = np.tile([1.0, 0.0, 0.0], (len(m.vertices), 1))
    pts, normals, colors = sample_surface(m, 500, np.random.default_rng(0))
    assert pts.shape == (500, 3) and np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-4)
    assert np.allclose(colors, [1, 0, 0])


@pytest.mark.parametrize("ext", [".glb", ".obj"])
def test_roundtrip(tmp_path, ext):
    m = taubin_smooth(surface_nets(sphere_field(20)))
    m.colors = np.random.default_rng(0).random((len(m.vertices), 3))
    path = tmp_path / f"m{ext}"
    (write_glb if ext == ".glb" else write_obj)(m, path)
    back = load_mesh(path)
    assert back.faces.shape == m.faces.shape and back.watertight
    assert np.allclose(back.vertices, m.vertices, atol=1e-5)
    assert np.allclose(back.colors, m.colors, atol=1e-3)


def test_stl_and_ply_loading(tmp_path):
    m = surface_nets(sphere_field(16))
    tri = m.vertices[m.faces].astype("<f4")
    data = b"\0" * 80 + len(tri).to_bytes(4, "little") + b"".join(
        np.zeros(3, "<f4").tobytes() + t.tobytes() + b"\0\0" for t in tri)
    (tmp_path / "m.stl").write_bytes(data)
    stl = load_mesh(tmp_path / "m.stl")
    assert stl.watertight and len(stl.faces) == len(m.faces)
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(m.vertices)}\nproperty float x\n"
              f"property float y\nproperty float z\nelement face {len(m.faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    body = "\n".join(" ".join(map(str, v)) for v in m.vertices) + "\n"
    body += "\n".join("3 " + " ".join(map(str, f)) for f in m.faces) + "\n"
    (tmp_path / "m.ply").write_text(header + body)
    assert load_mesh(tmp_path / "m.ply").watertight


def test_bad_files_are_rejected(tmp_path):
    (tmp_path / "x.glb").write_bytes(b"nope")
    with pytest.raises(ValueError):
        load_mesh(tmp_path / "x.glb")
    with pytest.raises(ValueError):
        load_mesh(tmp_path / "x.fbx")


def test_chamfer():
    a = np.random.default_rng(0).random((300, 3))
    assert chamfer_and_fscore(a, a) == (0.0, 1.0)
    c, f = chamfer_and_fscore(a, a + 5.0)
    assert c > 4 and f == 0.0


def test_connected_components_and_fragment_removal():
    from foxmesh_ai.geometry import connected_components, remove_small_components

    big = sphere_field(32, r=0.5, center=(-0.3, 0, 0))
    small = sphere_field(32, r=0.12, center=(0.65, 0, 0))
    m = surface_nets(np.maximum(big, small))
    assert connected_components(m).max() + 1 == 2
    cleaned = remove_small_components(m, 0.1)
    assert connected_components(cleaned).max() + 1 == 1 and cleaned.watertight
    assert remove_small_components(m, 0.01).faces.shape == m.faces.shape
