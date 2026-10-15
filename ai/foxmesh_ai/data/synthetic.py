"""Procedural training data: parametric shape families with Korean/English captions.

Shapes are built from signed distance functions, so every sample is a clean,
watertight, coloured mesh with known parts. This bootstraps training without
any external dataset; real scanned or modelled assets are added with
``Dataset.import_folder`` and mixed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..geometry import TriMesh, normalize, surface_nets, taubin_smooth

Sdf = Callable[[np.ndarray], np.ndarray]

# name -> (rgb, Korean adjective, English adjective)
PALETTE: dict[str, tuple[tuple[float, float, float], str, str]] = {
    "red": ((0.80, 0.12, 0.12), "빨간", "red"),
    "blue": ((0.15, 0.30, 0.80), "파란", "blue"),
    "green": ((0.15, 0.60, 0.20), "초록색", "green"),
    "yellow": ((0.95, 0.80, 0.15), "노란", "yellow"),
    "white": ((0.92, 0.92, 0.92), "흰", "white"),
    "black": ((0.08, 0.08, 0.08), "검정", "black"),
    "brown": ((0.45, 0.28, 0.12), "갈색", "brown"),
    "gray": ((0.55, 0.55, 0.58), "회색", "gray"),
    "gold": ((0.85, 0.65, 0.15), "금색", "golden"),
    "pink": ((0.95, 0.55, 0.70), "분홍색", "pink"),
    "orange": ((0.95, 0.50, 0.10), "주황색", "orange"),
    "purple": ((0.50, 0.20, 0.65), "보라색", "purple"),
}
SILVER = (0.75, 0.77, 0.80)
WOOD = (0.55, 0.36, 0.18)
DARK_WOOD = (0.32, 0.20, 0.10)


# ---------------------------------------------------------------------------
# SDF primitives (Y up). Approximate distances are fine: only the sign and the
# zero crossing matter for occupancy and surface extraction.
# ---------------------------------------------------------------------------


def sphere(c, r) -> Sdf:
    c = np.asarray(c, np.float64)
    return lambda p: np.linalg.norm(p - c, axis=-1) - r


def ellipsoid(c, radii) -> Sdf:
    c, radii = np.asarray(c, np.float64), np.asarray(radii, np.float64)
    def f(p):
        q = (p - c) / radii
        k = np.linalg.norm(q, axis=-1)
        return (k - 1) * radii.min()
    return f


def box(c, half, round_: float = 0.0) -> Sdf:
    c, half = np.asarray(c, np.float64), np.asarray(half, np.float64) - round_
    def f(p):
        q = np.abs(p - c) - half
        return (np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)) - round_
    return f


def cylinder(c, r, half_h, axis: int = 1) -> Sdf:
    c = np.asarray(c, np.float64)
    def f(p):
        q = p - c
        radial = np.linalg.norm(np.delete(q, axis, axis=-1), axis=-1) - r
        along = np.abs(q[..., axis]) - half_h
        d = np.stack([radial, along], -1)
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)
    return f


def frustum(base_c, r0, r1, h) -> Sdf:
    """Truncated cone along +Y from ``base_c``; r1 = 0 gives a cone."""
    base_c = np.asarray(base_c, np.float64)
    def f(p):
        q = p - base_c
        t = np.clip(q[..., 1] / h, 0, 1)
        radius = r0 + (r1 - r0) * t
        radial = np.linalg.norm(q[..., [0, 2]], axis=-1) - radius
        slab = np.maximum(-q[..., 1], q[..., 1] - h)
        return np.maximum(radial * np.cos(np.arctan2(abs(r0 - r1), h)), slab)
    return f


def torus(c, big_r, small_r, axis: int = 1) -> Sdf:
    c = np.asarray(c, np.float64)
    def f(p):
        q = p - c
        ring = np.linalg.norm(np.delete(q, axis, axis=-1), axis=-1) - big_r
        return np.sqrt(ring ** 2 + q[..., axis] ** 2) - small_r
    return f


def revolve(c, radius_fn: Callable[[np.ndarray], np.ndarray], y0: float, y1: float) -> Sdf:
    """Surface of revolution about the Y axis with radius(y) on [y0, y1] (relative to c)."""
    c = np.asarray(c, np.float64)
    def f(p):
        q = p - c
        y = q[..., 1]
        radial = np.linalg.norm(q[..., [0, 2]], axis=-1) - radius_fn(np.clip(y, y0, y1))
        return np.maximum(radial, np.maximum(y0 - y, y - y1))
    return f


def prism_roof(c, half_x, half_z, height) -> Sdf:
    """Gable roof: triangular prism along X with its base centred at ``c``."""
    c = np.asarray(c, np.float64)
    def f(p):
        q = p - c
        slope = height / half_z
        norm = np.sqrt(1 + slope ** 2)
        sides = (slope * np.abs(q[..., 2]) + q[..., 1] - height) / norm
        return np.maximum.reduce([sides, -q[..., 1], np.abs(q[..., 0]) - half_x])
    return f


def subtract(a: Sdf, b: Sdf) -> Sdf:
    return lambda p: np.maximum(a(p), -b(p))


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass
class Shape:
    family: str
    parts: list[tuple[Sdf, tuple[float, float, float]]]
    captions: list[str]
    tags: list[str]


def _color(rng) -> tuple[str, tuple[float, float, float], str, str]:
    name = rng.choice(list(PALETTE))
    rgb, ko, en = PALETTE[name]
    return name, rgb, ko, en


def _captions(rng, noun_ko: list[str], noun_en: list[str], color_ko: str, color_en: str,
              adj_ko: str = "", adj_en: str = "") -> list[str]:
    nk, ne = rng.choice(noun_ko), rng.choice(noun_en)
    ak = f"{adj_ko} " if adj_ko else ""
    ae = f"{adj_en} " if adj_en else ""
    return [
        f"{color_ko} {ak}{nk}",
        f"{ak}{nk} 만들어 줘",
        f"{color_ko} {nk} 3D 모델",
        f"a {color_en} {ae}{ne}",
        f"{ae}{ne}",
        f"3d model of a {ae}{color_en} {ne}",
    ]


def _sword(rng):
    _, rgb, ck, ce = _color(rng)
    length = rng.uniform(1.2, 1.7)
    width = rng.uniform(0.07, 0.14)
    guard = rng.uniform(0.3, 0.5)
    adj = ("긴", "long") if length > 1.55 else (("넓은", "broad") if width > 0.12 else ("", ""))
    blade_y0 = -0.45
    parts = [
        (box((0, blade_y0 + length / 2, 0), (width / 2, length / 2, 0.025)), SILVER),
        (frustum((0, blade_y0 + length, 0), width / 2 * 1.2, 0.0, 0.12), SILVER),
        (box((0, blade_y0, 0), (guard / 2, 0.035, 0.05)), rgb),
        (cylinder((0, blade_y0 - 0.17, 0), 0.035, 0.14), DARK_WOOD),
        (sphere((0, blade_y0 - 0.34, 0), 0.06), rgb),
    ]
    caps = _captions(rng, ["검", "칼", "장검"], ["sword", "blade", "longsword"], ck, ce, *adj)
    return Shape("sword", parts, caps, ["weapon", "game"])


def _table(rng):
    _, rgb, ck, ce = _color(rng)
    round_top = rng.random() < 0.35
    w, d, h = rng.uniform(1.0, 1.7), rng.uniform(0.6, 1.0), rng.uniform(0.6, 0.9)
    top_y = h / 2
    parts = []
    if round_top:
        r = min(w, d) / 2 + 0.15
        parts.append((cylinder((0, top_y, 0), r, 0.04), rgb))
        parts.append((cylinder((0, 0, 0), 0.06, h / 2), DARK_WOOD))
        parts.append((cylinder((0, -h / 2 + 0.03, 0), r * 0.5, 0.03), DARK_WOOD))
        adj = ("둥근", "round")
    else:
        parts.append((box((0, top_y, 0), (w / 2, 0.04, d / 2), 0.01), rgb))
        for sx in (-1, 1):
            for sz in (-1, 1):
                parts.append((box((sx * (w / 2 - 0.07), 0, sz * (d / 2 - 0.07)), (0.035, h / 2, 0.035)), DARK_WOOD))
        adj = ("긴", "long") if w > 1.5 else ("", "")
    caps = _captions(rng, ["테이블", "탁자", "책상"], ["table", "desk"], ck, ce, *adj)
    return Shape("table", parts, caps, ["furniture"])


def _chair(rng):
    _, rgb, ck, ce = _color(rng)
    s, seat_h, back_h = rng.uniform(0.7, 0.95), rng.uniform(0.75, 0.95), rng.uniform(0.6, 1.0)
    y0 = -0.8
    seat_y = y0 + seat_h
    parts = [(box((0, seat_y, 0), (s / 2, 0.04, s / 2), 0.01), rgb),
             (box((0, seat_y + back_h / 2, -s / 2 + 0.035), (s / 2, back_h / 2, 0.035), 0.01), rgb)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append((box((sx * (s / 2 - 0.05), y0 + seat_h / 2, sz * (s / 2 - 0.05)),
                              (0.03, seat_h / 2, 0.03)), DARK_WOOD))
    adj = ("등받이가 높은", "high-back") if back_h > 0.85 else ("", "")
    caps = _captions(rng, ["의자"], ["chair"], ck, ce, *adj)
    return Shape("chair", parts, caps, ["furniture"])


def _mug(rng):
    _, rgb, ck, ce = _color(rng)
    r, h = rng.uniform(0.35, 0.5), rng.uniform(0.7, 1.1)
    handle = rng.random() < 0.75
    body = subtract(cylinder((0, 0, 0), r, h / 2), cylinder((0, 0.08, 0), r - 0.06, h / 2))
    parts = [(body, rgb)]
    if handle:
        parts.append((torus((r + 0.12, 0, 0), 0.17, 0.045, axis=2), rgb))
        noun = (["머그컵", "머그", "컵"], ["mug", "coffee mug", "cup"])
    else:
        noun = (["컵", "잔"], ["cup", "glass"])
    adj = ("큰", "large") if h > 1.0 else ("", "")
    caps = _captions(rng, *noun, ck, ce, *adj)
    return Shape("mug", parts, caps, ["kitchen"])


def _tree(rng):
    pine = rng.random() < 0.6
    trunk_h = rng.uniform(0.35, 0.6)
    y0 = -0.9
    parts = [(cylinder((0, y0 + trunk_h / 2, 0), rng.uniform(0.07, 0.12), trunk_h / 2), WOOD)]
    leaf = PALETTE["green"][0]
    if pine:
        levels = int(rng.integers(2, 4))
        base, rad = y0 + trunk_h * 0.8, rng.uniform(0.5, 0.7)
        for i in range(levels):
            h = rng.uniform(0.5, 0.7)
            parts.append((frustum((0, base, 0), rad * (1 - 0.2 * i), 0.0, h), leaf))
            base += h * 0.55
        caps = _captions(rng, ["소나무", "침엽수", "나무"], ["pine tree", "fir tree", "tree"],
                         "초록색", "green", "키 큰" if levels == 3 else "", "tall" if levels == 3 else "")
    else:
        crown = rng.uniform(0.45, 0.65)
        parts.append((sphere((0, y0 + trunk_h + crown * 0.8, 0), crown), leaf))
        caps = _captions(rng, ["나무", "활엽수"], ["tree", "round tree"], "초록색", "green", "둥근", "round")
    return Shape("tree", parts, caps, ["nature"])


def _house(rng):
    _, rgb, ck, ce = _color(rng)
    w, d, h = rng.uniform(1.0, 1.5), rng.uniform(0.8, 1.2), rng.uniform(0.6, 0.9)
    roof_h = rng.uniform(0.35, 0.6)
    y0 = -0.7
    parts = [
        (box((0, y0 + h / 2, 0), (w / 2, h / 2, d / 2)), PALETTE["white"][0]),
        (prism_roof((0, y0 + h, 0), w / 2 + 0.06, d / 2 + 0.08, roof_h), rgb),
        (box((0, y0 + 0.22, d / 2 + 0.01), (0.12, 0.22, 0.03)), DARK_WOOD),
    ]
    if rng.random() < 0.5:
        parts.append((box((w / 4, y0 + h + roof_h * 0.6, 0), (0.07, 0.25, 0.07)), PALETTE["gray"][0]))
    caps = _captions(rng, ["집", "주택", "오두막"], ["house", "home", "cottage"], f"{ck} 지붕의", f"{ce}-roofed")
    return Shape("house", parts, caps, ["building"])


def _rocket(rng):
    _, rgb, ck, ce = _color(rng)
    r, h = rng.uniform(0.18, 0.28), rng.uniform(1.0, 1.4)
    y0 = -0.8
    parts = [(cylinder((0, y0 + h / 2, 0), r, h / 2), PALETTE["white"][0]),
             (frustum((0, y0 + h, 0), r, 0.0, rng.uniform(0.3, 0.5)), rgb)]
    fins = int(rng.integers(3, 5))
    for k in range(fins):
        a = 2 * np.pi * k / fins
        cx, cz = np.cos(a) * (r + 0.1), np.sin(a) * (r + 0.1)
        hx, hz = (0.12, 0.02) if abs(np.cos(a)) > 0.7 else (0.02, 0.12)
        parts.append((box((cx, y0 + 0.15, cz), (hx, 0.15, hz)), rgb))
    caps = _captions(rng, ["로켓", "우주선"], ["rocket", "spaceship"], ck, ce)
    return Shape("rocket", parts, caps, ["vehicle", "space"])


def _lamp(rng):
    _, rgb, ck, ce = _color(rng)
    pole = rng.uniform(0.8, 1.2)
    y0 = -0.85
    parts = [(cylinder((0, y0 + 0.03, 0), 0.25, 0.03), PALETTE["black"][0]),
             (cylinder((0, y0 + pole / 2, 0), 0.03, pole / 2), PALETTE["gray"][0]),
             (subtract(frustum((0, y0 + pole - 0.1, 0), 0.38, 0.2, 0.35),
                       frustum((0, y0 + pole - 0.12, 0), 0.34, 0.17, 0.38)), rgb)]
    caps = _captions(rng, ["스탠드", "램프", "조명"], ["lamp", "floor lamp", "desk lamp"], ck, ce)
    return Shape("lamp", parts, caps, ["furniture", "light"])


def _bottle(rng):
    _, rgb, ck, ce = _color(rng)
    r, h, neck = rng.uniform(0.25, 0.35), rng.uniform(0.9, 1.2), rng.uniform(0.25, 0.4)
    y0 = -0.85
    def radius(y):
        t = (y - h * 0.6) / (h * 0.2)
        return np.where(y < h * 0.6, r, np.where(y < h * 0.8, r + (0.09 - r) * np.clip(t, 0, 1), 0.09))
    parts = [(revolve((0, y0, 0), radius, 0, h + neck - 0.1), rgb),
             (cylinder((0, y0 + h + neck - 0.05, 0), 0.1, 0.06), PALETTE["black"][0])]
    caps = _captions(rng, ["병", "물병", "와인병"], ["bottle", "wine bottle"], ck, ce)
    return Shape("bottle", parts, caps, ["kitchen"])


def _mushroom(rng):
    _, rgb, ck, ce = _color(rng)
    stem_h, cap_r = rng.uniform(0.5, 0.8), rng.uniform(0.5, 0.75)
    y0 = -0.75
    cap = lambda p, c=np.array([0, y0 + stem_h, 0]): np.maximum(
        ellipsoid(c, (cap_r, cap_r * 0.7, cap_r))(p), (c[1] - 0.02) - p[..., 1])
    parts = [(cylinder((0, y0 + stem_h / 2, 0), 0.14, stem_h / 2), PALETTE["white"][0]), (cap, rgb)]
    caps = _captions(rng, ["버섯"], ["mushroom"], ck, ce)
    return Shape("mushroom", parts, caps, ["nature", "game"])


def _snowman(rng):
    _, rgb, ck, ce = _color(rng)
    r0, r1, r2 = rng.uniform(0.38, 0.45), rng.uniform(0.28, 0.33), rng.uniform(0.18, 0.23)
    y0 = -0.85
    c0, c1 = y0 + r0, y0 + 2 * r0 + r1 * 0.8
    c2 = c1 + r1 + r2 * 0.8
    white = PALETTE["white"][0]
    parts = [(sphere((0, c0, 0), r0), white), (sphere((0, c1, 0), r1), white),
             (sphere((0, c2, 0), r2), white),
             (cylinder((0, c2 + r2 * 0.9, 0), r2 * 0.75, 0.1), rgb)]
    caps = _captions(rng, ["눈사람"], ["snowman"], f"{ck} 모자를 쓴", f"{ce}-hatted")
    return Shape("snowman", parts, caps, ["character", "winter"])


def _donut(rng):
    _, rgb, ck, ce = _color(rng)
    big, small = rng.uniform(0.45, 0.55), rng.uniform(0.18, 0.25)
    dough = torus((0, 0, 0), big, small)
    icing = lambda p: np.maximum(torus((0, 0.02, 0), big, small + 0.02)(p), -p[..., 1] + 0.03)
    parts = [(dough, (0.80, 0.55, 0.30)), (icing, rgb)]
    caps = _captions(rng, ["도넛"], ["donut", "doughnut"], f"{ck} 아이싱", f"{ce}-iced")
    return Shape("donut", parts, caps, ["food"])


def _dumbbell(rng):
    _, rgb, ck, ce = _color(rng)
    length, disc = rng.uniform(1.2, 1.6), rng.uniform(0.25, 0.35)
    parts = [(cylinder((0, 0, 0), 0.05, length / 2, axis=0), PALETTE["gray"][0])]
    for sx in (-1, 1):
        parts.append((cylinder((sx * (length / 2 - 0.12), 0, 0), disc, 0.09, axis=0), rgb))
    caps = _captions(rng, ["아령", "덤벨"], ["dumbbell"], ck, ce)
    return Shape("dumbbell", parts, caps, ["sports"])


def _barrel(rng):
    r, h = rng.uniform(0.4, 0.55), rng.uniform(0.9, 1.3)
    bulge = rng.uniform(0.05, 0.12)
    radius = lambda y: r + bulge * (1 - (2 * y / h) ** 2)
    parts = [(revolve((0, 0, 0), radius, -h / 2, h / 2), WOOD)]
    for y in (-h / 3, h / 3):
        parts.append((torus((0, y, 0), radius(np.array(y)) + 0.005, 0.025), PALETTE["gray"][0]))
    caps = _captions(rng, ["나무통", "통", "술통"], ["barrel", "wooden barrel"], "갈색", "brown")
    return Shape("barrel", parts, caps, ["game", "prop"])


def _tower(rng):
    _, rgb, ck, ce = _color(rng)
    r, h = rng.uniform(0.3, 0.42), rng.uniform(1.0, 1.4)
    y0 = -0.9
    stone = PALETTE["gray"][0]
    parts = [(cylinder((0, y0 + h / 2, 0), r, h / 2), stone)]
    if rng.random() < 0.5:
        parts.append((frustum((0, y0 + h, 0), r + 0.08, 0.0, rng.uniform(0.4, 0.6)), rgb))
        adj = ("뾰족한", "pointed")
    else:
        for k in range(8):
            a = 2 * np.pi * k / 8
            parts.append((box((np.cos(a) * r * 0.85, y0 + h + 0.07, np.sin(a) * r * 0.85),
                              (0.06, 0.07, 0.06)), stone))
        adj = ("성곽", "castle")
    caps = _captions(rng, ["탑", "성탑", "망루"], ["tower", "castle tower"], ck, ce, *adj)
    return Shape("tower", parts, caps, ["building", "game"])


def _vase(rng):
    _, rgb, ck, ce = _color(rng)
    h = rng.uniform(1.0, 1.4)
    belly, mouth = rng.uniform(0.3, 0.45), rng.uniform(0.12, 0.2)
    def radius(y):
        t = y / h
        return 0.12 + belly * np.sin(np.pi * np.clip(t * 1.1, 0, 1)) + (mouth - 0.12) * t ** 4
    outer = revolve((0, -h / 2, 0), radius, 0, h)
    inner = revolve((0, -h / 2, 0), lambda y: np.maximum(radius(y) - 0.05, 0.02), 0.1, h + 0.2)
    parts = [(subtract(outer, inner), rgb)]
    caps = _captions(rng, ["꽃병", "화병", "항아리"], ["vase", "pot"], ck, ce)
    return Shape("vase", parts, caps, ["decor"])


FAMILIES: dict[str, Callable[[np.random.Generator], Shape]] = {
    "sword": _sword, "table": _table, "chair": _chair, "mug": _mug, "tree": _tree,
    "house": _house, "rocket": _rocket, "lamp": _lamp, "bottle": _bottle,
    "mushroom": _mushroom, "snowman": _snowman, "donut": _donut, "dumbbell": _dumbbell,
    "barrel": _barrel, "tower": _tower, "vase": _vase,
}


def shape_to_mesh(shape: Shape, res: int = 64, smooth: int = 4) -> TriMesh:
    """Evaluate the union of the parts on a grid and extract a coloured, normalised mesh."""
    c = (np.arange(res) + 0.5) / res * 2 - 1
    grid = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    dists = np.stack([sdf(grid) for sdf, _ in shape.parts], -1)
    field = (-dists.min(-1)).reshape(res, res, res)
    mesh = surface_nets(field)
    if smooth:
        mesh = taubin_smooth(mesh, smooth)
    part = np.stack([sdf(mesh.vertices.astype(np.float64)) for sdf, _ in shape.parts], -1).argmin(-1)
    palette = np.array([rgb for _, rgb in shape.parts], np.float32)
    mesh.colors = palette[part]
    mesh, _ = normalize(mesh)
    return mesh


def generate_shape(rng: np.random.Generator, family: str | None = None) -> Shape:
    name = family or rng.choice(list(FAMILIES))
    return FAMILIES[name](rng)


def build_synthetic_dataset(root, count: int, seed: int = 0, res: int = 64, name: str = "foxmesh-synthetic",
                            families: list[str] | None = None, progress=None):
    """Create (or extend) a dataset with ``count`` procedural shapes, balanced over families."""
    from .dataset import Dataset

    dataset = Dataset.create(root, name, description="Procedural shapes with Korean/English captions",
                             license="CC0-1.0", exist_ok=True)
    rng = np.random.default_rng(seed)
    names = families or list(FAMILIES)
    added = 0
    for n in range(count):
        family = names[n % len(names)]
        shape = generate_shape(rng, family)
        mesh = shape_to_mesh(shape, res=res)
        _, new = dataset.add(mesh, shape.captions, tags=[family, *shape.tags], source="synthetic",
                             license="CC0-1.0", extra={"family": family}, save=False)
        added += new
        if progress:
            progress(n + 1, count)
    dataset.save()
    return dataset, added
