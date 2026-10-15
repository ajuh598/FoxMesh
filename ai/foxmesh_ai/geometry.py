"""Mesh geometry in NumPy: file I/O, normalisation, sampling, voxelisation, surface extraction.

Coordinates follow glTF/OBJ convention (Y up). Training space is the cube
[-1, 1]^3; meshes are normalised to fit inside [-NORMALIZED_EXTENT, NORMALIZED_EXTENT].
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NORMALIZED_EXTENT = 0.9


@dataclass
class TriMesh:
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int64, counter-clockwise seen from outside
    colors: np.ndarray | None = None  # (N, 3) float32 in [0, 1]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, np.float32).reshape(-1, 3)
        self.faces = np.asarray(self.faces, np.int64).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.clip(np.asarray(self.colors, np.float32).reshape(-1, 3), 0, 1)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def face_areas(self) -> np.ndarray:
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def volume(self) -> float:
        """Signed volume (positive for closed, outward-facing meshes)."""
        a, b, c = (self.vertices[self.faces[:, i]].astype(np.float64) for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def edge_manifold_stats(self) -> tuple[int, int]:
        """Return (boundary_edges, non_manifold_edges); both 0 for a watertight mesh."""
        edges = np.sort(np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                                        self.faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int((counts == 1).sum()), int((counts > 2).sum())

    @property
    def watertight(self) -> bool:
        return len(self.faces) > 0 and self.edge_manifold_stats() == (0, 0)

    def copy(self) -> "TriMesh":
        return TriMesh(self.vertices.copy(), self.faces.copy(),
                       None if self.colors is None else self.colors.copy(), dict(self.meta))


# ---------------------------------------------------------------------------
# Normalisation and sampling
# ---------------------------------------------------------------------------


def normalize(mesh: TriMesh, extent: float = NORMALIZED_EXTENT) -> tuple[TriMesh, dict]:
    """Centre the bounding box at the origin and scale the largest side to ``2 * extent``."""
    lo, hi = mesh.bounds
    center = (lo + hi) / 2
    scale = (2 * extent) / max(float((hi - lo).max()), 1e-9)
    out = mesh.copy()
    out.vertices = ((mesh.vertices - center) * scale).astype(np.float32)
    return out, {"center": center.tolist(), "scale": scale}


def sample_surface(mesh: TriMesh, count: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Area-weighted surface samples: (points, normals, colors or None)."""
    areas = mesh.face_areas()
    if areas.sum() <= 0:
        raise ValueError("mesh has no surface area")
    face_idx = rng.choice(len(areas), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = 1 - u - v
    tri = mesh.faces[face_idx]
    a, b, c = (mesh.vertices[tri[:, i]] for i in range(3))
    points = a * w[:, None] + b * u[:, None] + c * v[:, None]
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    colors = None
    if mesh.colors is not None:
        ca, cb, cc = (mesh.colors[tri[:, i]] for i in range(3))
        colors = ca * w[:, None] + cb * u[:, None] + cc * v[:, None]
    return points.astype(np.float32), normals.astype(np.float32), colors


# ---------------------------------------------------------------------------
# Voxelisation (inside/outside) by ray parity
# ---------------------------------------------------------------------------


def _parity_axis(mesh: TriMesh, res: int, axis: int) -> np.ndarray:
    """Cast rays along ``axis`` through every column centre and count crossings (vectorised).

    Every (triangle, column) pair inside the triangle's bounding box is tested at
    once; a voxel is inside when an odd number of crossings lie below its centre.
    """
    others = [a for a in range(3) if a != axis]
    v = mesh.vertices.astype(np.float64)
    tri = v[mesh.faces]
    p = tri[:, :, others]  # (M, 3, 2) projected onto the ray plane
    depth = tri[:, :, axis]
    # Column indices whose centres fall inside each triangle's bounding box.
    lo = np.ceil((p.min(axis=1) + 1) / 2 * res - 0.5).astype(np.int64)
    hi = np.floor((p.max(axis=1) + 1) / 2 * res - 0.5).astype(np.int64)
    lo, hi = np.clip(lo, 0, res - 1), np.clip(hi, -1, res - 1)
    span = np.maximum(hi - lo + 1, 0)
    count = span[:, 0] * span[:, 1]
    keep = count > 0
    tri_idx = np.repeat(np.nonzero(keep)[0], count[keep])
    if len(tri_idx) == 0:
        return np.zeros((res, res, res), bool)
    starts = np.repeat(np.cumsum(count[keep]) - count[keep], count[keep])
    k = np.arange(len(tri_idx)) - starts
    width = span[tri_idx, 1]
    ii = lo[tri_idx, 0] + k // width
    jj = lo[tri_idx, 1] + k % width
    cx = (ii + 0.5) / res * 2 - 1
    cy = (jj + 0.5) / res * 2 - 1
    (ax, ay), (bx, by), (qx, qy) = (p[tri_idx, i].T for i in range(3))
    det = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
    ok = np.abs(det) > 1e-18
    det = np.where(ok, det, 1.0)
    l1 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / det
    l2 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / det
    l3 = 1 - l1 - l2
    # Half-open edge rule so a ray through a shared edge is counted once.
    hit = ok & (l1 >= 0) & (l2 >= 0) & (l3 > 0)
    d = depth[tri_idx]
    z = l1 * d[:, 0] + l2 * d[:, 1] + l3 * d[:, 2]
    ii, jj, z = ii[hit], jj[hit], z[hit]
    # Index of the first voxel centre above each crossing; parity of the running count.
    kz = np.clip(np.ceil((z + 1) / 2 * res - 0.5).astype(np.int64), 0, res)
    diff = np.zeros((res, res, res + 1), np.int32)
    np.add.at(diff, (ii, jj, kz), 1)
    inside = (np.cumsum(diff, axis=2)[:, :, :res] % 2) == 1
    # inside is indexed [others[0], others[1], axis]; reorder to [x, y, z].
    order = np.argsort([others[0], others[1], axis])
    return np.transpose(inside, order)


def voxelize(mesh: TriMesh, res: int = 64) -> np.ndarray:
    """Occupancy grid over [-1, 1]^3, indexed [x, y, z].

    Parity is computed along all three axes and combined by majority vote, which
    tolerates small holes and duplicated faces in imperfect meshes.
    """
    votes = sum(_parity_axis(mesh, res, axis).astype(np.uint8) for axis in range(3))
    return votes >= 2


def grid_lookup(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-voxel occupancy for points in [-1, 1]^3."""
    res = grid.shape[0]
    idx = np.clip(((points + 1) / 2 * res).astype(int), 0, res - 1)
    return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


# ---------------------------------------------------------------------------
# Surface extraction (naive surface nets) and smoothing
# ---------------------------------------------------------------------------


_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])  # index 4i+2j+k
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1]


def make_well_composed(inside: np.ndarray, max_iterations: int = 64) -> np.ndarray:
    """Add inside samples until no two samples touch only along an edge or at a corner.

    Such "critical" configurations are where dual contouring glues separate surface
    sheets together (non-manifold edges). Removing them, always by filling a sample,
    makes every 2x2x2 block have one connected inside and one connected outside part,
    so the extracted surface is a manifold. Changes are local and only thicken thin
    features or close one-sample gaps.
    """
    g = inside.copy()
    n = g.shape
    for _ in range(max_iterations):
        changed = False
        # Face-diagonal pairs in each of the three square orientations.
        for u, v in ((0, 1), (1, 2), (0, 2)):
            def view(du, dv):
                idx = [slice(None)] * 3
                idx[u] = slice(du, n[u] - 1 + du)
                idx[v] = slice(dv, n[v] - 1 + dv)
                return tuple(idx)
            p00, p10, p01, p11 = (g[view(a, b)] for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)))
            bad1 = p00 & p11 & ~p10 & ~p01
            bad2 = p10 & p01 & ~p00 & ~p11
            if bad1.any() or bad2.any():
                g[view(1, 0)] |= bad1
                g[view(0, 0)] |= bad2
                changed = True
        # Body-diagonal pairs isolated within a 2x2x2 block (inside or outside).
        corners = [g[c[0]:n[0] - 1 + c[0], c[1]:n[1] - 1 + c[1], c[2]:n[2] - 1 + c[2]] for c in _CORNERS]
        total = sum(c.astype(np.uint8) for c in corners)
        for a in range(4):
            b = 7 - a  # opposite corner
            lone_in = corners[a] & corners[b] & (total == 2)
            lone_out = ~corners[a] & ~corners[b] & (total == 6)
            fix = lone_in | lone_out
            if fix.any():
                # Fill a neighbour of corner ``a`` (inside pair) or corner ``a`` itself (outside pair).
                ta = _CORNERS[a]
                tn = _CORNERS[a ^ 1]
                g[ta[0]:n[0] - 1 + ta[0], ta[1]:n[1] - 1 + ta[1], ta[2]:n[2] - 1 + ta[2]] |= lone_out
                g[tn[0]:n[0] - 1 + tn[0], tn[1]:n[1] - 1 + tn[1], tn[2]:n[2] - 1 + tn[2]] |= lone_in
                changed = True
        if not changed:
            return g
    return g


def surface_nets(field: np.ndarray, level: float = 0.0, bounds: float = 1.0) -> TriMesh:
    """Extract a closed, manifold mesh from a scalar field on a grid over [-bounds, bounds]^3.

    Values greater than ``level`` are inside. The field is padded with outside
    values, so the result is always closed, and the inside set is made
    well-composed first (see ``make_well_composed``), so no edge is shared by more
    than two faces. Faces are oriented outwards.
    """
    f = np.pad(field.astype(np.float32) - level, 1, constant_values=-1.0)
    inside = make_well_composed(f > 0)
    # Samples flipped to inside get a small positive value so crossings sit next to them.
    eps = 1e-3 * max(float(np.abs(f).max()), 1e-6)
    f = np.where(inside & (f <= 0), eps, f)
    f = np.where(~inside & (f > 0), -eps, f)
    n = np.array(f.shape)
    cell_shape = tuple(n - 1)
    corner_vals = np.stack([f[c[0]:c[0] + cell_shape[0], c[1]:c[1] + cell_shape[1],
                              c[2]:c[2] + cell_shape[2]] for c in _CORNERS], axis=-1)
    corner_in = corner_vals > 0
    active = corner_in.any(axis=-1) & ~corner_in.all(axis=-1)
    cells = np.argwhere(active)
    vals = corner_vals[active]

    acc = np.zeros((len(cells), 3), np.float64)
    cnt = np.zeros(len(cells), np.float64)
    for a, b in _EDGES:
        va, vb = vals[:, a], vals[:, b]
        cross = (va > 0) != (vb > 0)
        t = np.where(cross, va / np.where(cross, va - vb, 1.0), 0.0)
        point = _CORNERS[a] + t[:, None] * (_CORNERS[b] - _CORNERS[a])
        acc += np.where(cross[:, None], point, 0.0)
        cnt += cross
    verts_grid = cells + acc / np.maximum(cnt, 1)[:, None]

    index = -np.ones(cell_shape, np.int64)
    index[tuple(cells.T)] = np.arange(len(cells))

    faces = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        step = np.zeros(3, int)
        step[axis] = 1
        lo = inside[tuple(slice(0, n[d] - step[d]) for d in range(3))]
        hi = inside[tuple(slice(step[d], n[d]) for d in range(3))]
        pts = np.argwhere(lo != hi)
        pts = pts[(pts[:, b] >= 1) & (pts[:, c] >= 1)]
        if len(pts) == 0:
            continue
        ring = []
        for db, dc in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            offset = np.zeros(3, int)
            offset[b], offset[c] = db, dc
            q = pts + offset
            ring.append(index[q[:, 0], q[:, 1], q[:, 2]])
        ring = np.stack(ring, axis=1)
        outward_positive = lo[tuple(pts.T)]  # inside -> outside along +axis
        faces.append(np.where(outward_positive[:, None], ring, ring[:, ::-1]))
    if not faces:
        return TriMesh(np.zeros((0, 3)), np.zeros((0, 3)))
    quads = np.concatenate(faces)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    res = field.shape[0]
    vertices = (verts_grid - 1 + 0.5) / res * 2 * bounds - bounds  # undo padding, to world
    return TriMesh(vertices, tris)


def connected_components(mesh: TriMesh) -> np.ndarray:
    """Component label per face (faces sharing a vertex are connected)."""
    parent = np.arange(len(mesh.vertices))

    def find(x: np.ndarray) -> np.ndarray:
        while True:
            p = parent[x]
            if np.array_equal(p, x):
                return x
            parent[x] = parent[p]
            x = parent[x]

    for a, b in ((0, 1), (1, 2)):
        u, v = mesh.faces[:, a], mesh.faces[:, b]
        while True:  # vectorised union: hook roots together until every edge agrees
            ru, rv = find(u.copy()), find(v.copy())
            diff = ru != rv
            if not diff.any():
                break
            lo, hi = np.minimum(ru[diff], rv[diff]), np.maximum(ru[diff], rv[diff])
            parent[hi] = lo
    roots = find(np.arange(len(mesh.vertices)))
    return np.unique(roots[mesh.faces[:, 0]], return_inverse=True)[1]


def remove_small_components(mesh: TriMesh, min_fraction: float = 0.02) -> TriMesh:
    """Drop disconnected pieces whose surface area is below ``min_fraction`` of the total."""
    labels = connected_components(mesh)
    area = np.bincount(labels, weights=mesh.face_areas())
    keep_label = area >= min_fraction * area.sum()
    keep_faces = keep_label[labels]
    if keep_faces.all():
        return mesh
    faces = mesh.faces[keep_faces]
    used = np.unique(faces)
    remap = -np.ones(len(mesh.vertices), np.int64)
    remap[used] = np.arange(len(used))
    return TriMesh(mesh.vertices[used], remap[faces],
                   None if mesh.colors is None else mesh.colors[used], dict(mesh.meta))


def taubin_smooth(mesh: TriMesh, iterations: int = 6, lam: float = 0.5, mu: float = -0.53) -> TriMesh:
    """Volume-preserving Laplacian smoothing (Taubin's lambda/mu)."""
    n = len(mesh.vertices)
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    v = mesh.vertices.astype(np.float64)
    for _ in range(iterations):
        for factor in (lam, mu):
            acc = np.zeros_like(v)
            np.add.at(acc, edges[:, 0], v[edges[:, 1]])
            np.add.at(acc, edges[:, 1], v[edges[:, 0]])
            lap = acc / np.maximum(degree, 1)[:, None] - v
            v = v + factor * lap
    out = mesh.copy()
    out.vertices = v.astype(np.float32)
    return out


def chamfer_and_fscore(a: np.ndarray, b: np.ndarray, threshold: float = 0.02,
                       chunk: int = 1024) -> tuple[float, float]:
    """Symmetric L1-Chamfer distance (mean of both directions) and F-score at ``threshold``."""
    def nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        out = np.empty(len(src))
        for i in range(0, len(src), chunk):
            d = ((src[i:i + chunk, None, :] - dst[None, :, :]) ** 2).sum(-1)
            out[i:i + chunk] = np.sqrt(d.min(axis=1))
        return out
    da, db = nearest(a, b), nearest(b, a)
    precision = (da < threshold).mean()
    recall = (db < threshold).mean()
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float((da.mean() + db.mean()) / 2), float(f)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _triangulate(polys: list[list[int]]) -> list[list[int]]:
    return [[p[0], p[k], p[k + 1]] for p in polys for k in range(1, len(p) - 1)]


def _load_obj(path: Path) -> TriMesh:
    verts, colors, polys = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            parts = [float(x) for x in line.split()[1:]]
            verts.append(parts[:3])
            if len(parts) >= 6:
                colors.append(parts[3:6])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            polys.append([i - 1 if i > 0 else len(verts) + i for i in idx])
    cols = np.array(colors) if colors and len(colors) == len(verts) else None
    return TriMesh(np.array(verts), np.array(_triangulate(polys)), cols)


def _load_stl(path: Path) -> TriMesh:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                            count=count, offset=84)
        tri = rec["v"].reshape(-1, 3, 3)
    else:
        vals = [list(map(float, line.split()[1:4])) for line in data.decode(errors="replace").splitlines()
                if line.strip().startswith("vertex")]
        tri = np.array(vals).reshape(-1, 3, 3)
    flat = tri.reshape(-1, 3)
    uniq, inverse = np.unique(flat.round(6), axis=0, return_inverse=True)
    return TriMesh(uniq, inverse.reshape(-1, 3))


def _load_ply(path: Path) -> TriMesh:
    data = path.read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end = data.index(b"\n", end) + 1
    header = data[:end].decode().splitlines()
    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    elements: list[tuple[str, int, list[list[str]]]] = []
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            elements.append((parts[1], int(parts[2]), []))
        elif parts[:1] == ["property"]:
            elements[-1][2].append(parts[1:])
    types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
             "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2",
             "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
    verts = colors = None
    faces: list[list[int]] = []
    if fmt == "ascii":
        rows = data[end:].decode().split("\n")
        pos = 0
        for name, count, props in elements:
            chunk = rows[pos:pos + count]
            pos += count
            if name == "vertex":
                names = [p[-1] for p in props]
                arr = np.array([list(map(float, r.split())) for r in chunk])
                verts = arr[:, [names.index(a) for a in "xyz"]]
                if {"red", "green", "blue"} <= set(names):
                    colors = arr[:, [names.index(c) for c in ("red", "green", "blue")]] / 255
            elif name == "face":
                faces = [list(map(int, r.split()))[1:] for r in chunk]
    else:
        endian = "<" if "little" in fmt else ">"
        offset = end
        for name, count, props in elements:
            if name == "face" or any(p[0] == "list" for p in props):
                count_t, index_t = types[props[0][1]], types[props[0][2]]
                for _ in range(count):
                    (k,) = np.frombuffer(data, endian + count_t, 1, offset)
                    offset += np.dtype(count_t).itemsize
                    idx = np.frombuffer(data, endian + index_t, int(k), offset)
                    offset += int(k) * np.dtype(index_t).itemsize
                    if name == "face":
                        faces.append(idx.tolist())
            else:
                dtype = np.dtype([(p[-1], endian + types[p[0]]) for p in props])
                arr = np.frombuffer(data, dtype, count, offset)
                offset += count * dtype.itemsize
                if name == "vertex":
                    verts = np.stack([arr[a] for a in "xyz"], axis=1)
                    if {"red", "green", "blue"} <= set(arr.dtype.names):
                        colors = np.stack([arr[c] for c in ("red", "green", "blue")], axis=1) / 255
    return TriMesh(verts, np.array(_triangulate(faces)), colors)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    s = np.array(node.get("scale", [1, 1, 1]), np.float64)
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def _load_glb(path: Path) -> TriMesh:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError("not a binary glTF file")
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len + 8
    sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    dtypes = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(i: int) -> np.ndarray:
        acc = gltf["accessors"][i]
        view = gltf["bufferViews"][acc["bufferView"]]
        width = widths[acc["type"]]
        offset = bin_start + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride", 0)
        item = sizes[acc["componentType"]] * width
        if stride and stride != item:
            rows = [np.frombuffer(data, dtypes[acc["componentType"]], width, offset + r * stride)
                    for r in range(acc["count"])]
            arr = np.stack(rows)
        else:
            arr = np.frombuffer(data, dtypes[acc["componentType"]], acc["count"] * width, offset)
            arr = arr.reshape(acc["count"], width)
        if acc.get("normalized") and acc["componentType"] in (5121, 5123):
            arr = arr / (255.0 if acc["componentType"] == 5121 else 65535.0)
        return arr.astype(np.float64) if arr.dtype.kind == "f" or acc.get("normalized") else arr

    all_v, all_f, all_c = [], [], []
    has_colors = True

    def visit(node_index: int, parent: np.ndarray) -> None:
        nonlocal has_colors
        node = gltf["nodes"][node_index]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = accessor(prim["attributes"]["POSITION"])
                pos = pos @ m[:3, :3].T + m[:3, 3]
                idx = (accessor(prim["indices"]).reshape(-1) if "indices" in prim
                       else np.arange(len(pos)))
                base = sum(len(v) for v in all_v)
                all_v.append(pos)
                all_f.append(idx.reshape(-1, 3).astype(np.int64) + base)
                if "COLOR_0" in prim["attributes"]:
                    all_c.append(accessor(prim["attributes"]["COLOR_0"])[:, :3])
                else:
                    has_colors = False
        for child in node.get("children", []):
            visit(child, m)

    scene = gltf.get("scenes", [{"nodes": list(range(len(gltf.get("nodes", []))))}])[gltf.get("scene", 0)]
    for root in scene["nodes"]:
        visit(root, np.eye(4))
    if not all_v:
        raise ValueError("no triangle meshes in file")
    colors = np.concatenate(all_c) if has_colors and all_c else None
    return TriMesh(np.concatenate(all_v), np.concatenate(all_f), colors)


LOADERS = {".obj": _load_obj, ".stl": _load_stl, ".ply": _load_ply, ".glb": _load_glb}
MESH_EXTENSIONS = tuple(LOADERS)


def load_mesh(path: str | Path) -> TriMesh:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported mesh format {path.suffix}")
    mesh = loader(path)
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh has non-finite vertices")
    if mesh.faces.max() >= len(mesh.vertices) or mesh.faces.min() < 0:
        raise ValueError("face index out of range")
    return mesh


def vertex_normals(mesh: TriMesh) -> np.ndarray:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    fn = np.cross(b - a, c - a)
    acc = np.zeros_like(mesh.vertices, dtype=np.float64)
    for i in range(3):
        np.add.at(acc, mesh.faces[:, i], fn)
    return (acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def write_obj(mesh: TriMesh, path: str | Path) -> None:
    lines = ["# FoxMesh AI"]
    if mesh.colors is not None:
        lines += [f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}"
                  for (x, y, z), (r, g, b) in zip(mesh.vertices, mesh.colors)]
    else:
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    Path(path).write_text("\n".join(lines) + "\n")


def write_glb(mesh: TriMesh, path: str | Path) -> None:
    """Binary glTF with smooth normals and optional vertex colours (COLOR_0)."""
    chunks: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def add(arr: np.ndarray, kind: str, ctype: int, target: int, **extra) -> int:
        raw = arr.tobytes()
        offset = sum(len(c) for c in chunks)
        chunks.append(raw + b"\0" * (-len(raw) % 4))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        count = arr.shape[0] if kind != "SCALAR" else arr.size
        accessors.append({"bufferView": len(views) - 1, "componentType": ctype, "count": int(count),
                          "type": kind, **extra})
        return len(accessors) - 1

    v = mesh.vertices.astype("<f4")
    attributes = {
        "POSITION": add(v, "VEC3", 5126, 34962, min=v.min(0).tolist(), max=v.max(0).tolist()),
        "NORMAL": add(vertex_normals(mesh).astype("<f4"), "VEC3", 5126, 34962),
    }
    if mesh.colors is not None:
        attributes["COLOR_0"] = add(mesh.colors.astype("<f4"), "VEC3", 5126, 34962)
    indices = add(mesh.faces.astype("<u4").reshape(-1), "SCALAR", 5125, 34963)
    material = {"name": "FoxMesh", "pbrMetallicRoughness": {
        "baseColorFactor": [1, 1, 1, 1] if mesh.colors is not None else [0.62, 0.62, 0.65, 1],
        "metallicFactor": 0.0, "roughnessFactor": 0.6}}
    bin_chunk = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "FoxMesh AI"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "foxmesh"}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": indices,
                                    "material": 0, "mode": 4}]}],
        "materials": [material],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": views, "accessors": accessors,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(bin_chunk)))
        fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fh.write(struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)


def write_mesh(mesh: TriMesh, path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".glb":
        write_glb(mesh, path)
    elif suffix == ".obj":
        write_obj(mesh, path)
    else:
        raise ValueError(f"cannot write {suffix}")
