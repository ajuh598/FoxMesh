"""CPU procedural generator: builds simple closed meshes from keyword recipes.

It exists so the whole pipeline (routing -> generation -> Blender -> validation)
runs end to end on any machine, and serves as a last-resort fallback model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from ..schemas import ModelSpec, OutputFormat, TaskSpec
from .base import GenerationAdapter, GenerationResult
from .mesh import Mesh, Vec

__all__ = ["Mesh", "ProceduralAdapter", "box", "frustum", "generate_mesh", "sphere"]

def box(size: Vec, center: Vec = (0, 0, 0)) -> Mesh:
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    v = [(cx + x * sx, cy + y * sy, cz + z * sz)
         for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    # Index = 4*x + 2*y + z with x, y, z in {0, 1}.
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return Mesh(v, faces)


def frustum(r_bottom: float, r_top: float, height: float, segments: int = 24,
            center: Vec = (0, 0, 0)) -> Mesh:
    """Closed cylinder/cone along Z. ``r_top == 0`` gives a cone."""
    cx, cy, cz = center
    z0, z1 = cz - height / 2, cz + height / 2
    n = segments

    def ring(r: float, z: float) -> list[Vec]:
        return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), z)
                for i in range(n)]

    bottom_cap = tuple(reversed(range(n)))  # facing -Z
    if r_top == 0:
        verts = ring(r_bottom, z0) + [(cx, cy, z1)]
        faces = [(i, (i + 1) % n, n) for i in range(n)] + [bottom_cap]
    else:
        verts = ring(r_bottom, z0) + ring(r_top, z1)
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        faces += [bottom_cap, tuple(range(n, 2 * n))]
    return Mesh(verts, faces)


def sphere(radius: float, rings: int = 12, segments: int = 24, center: Vec = (0, 0, 0)) -> Mesh:
    cx, cy, cz = center
    verts: list[Vec] = [(cx, cy, cz - radius)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings - math.pi / 2
        for i in range(segments):
            a = 2 * math.pi * i / segments
            verts.append((cx + radius * math.cos(phi) * math.cos(a),
                          cy + radius * math.cos(phi) * math.sin(a),
                          cz + radius * math.sin(phi)))
    verts.append((cx, cy, cz + radius))
    top = len(verts) - 1
    faces: list[tuple[int, ...]] = []
    for i in range(segments):
        faces.append((0, 1 + (i + 1) % segments, 1 + i))
    for ring in range(rings - 2):
        base = 1 + ring * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((base + i, base + j, base + segments + j, base + segments + i))
    last = 1 + (rings - 2) * segments
    for i in range(segments):
        faces.append((last + i, last + (i + 1) % segments, top))
    return Mesh(verts, faces)


# ---------------------------------------------------------------------------
# Recipes: keyword -> assembled object. Units are metres, Z up.
# ---------------------------------------------------------------------------


def _sword(seg: int) -> Mesh:
    m = Mesh()
    m.add(box((0.06, 0.012, 0.75), (0, 0, 0.495)), "blade")
    m.add(frustum(0.03, 0.0, 0.08, 4, (0, 0, 0.91)), "tip")
    m.add(box((0.22, 0.03, 0.03), (0, 0, 0.105)), "guard")
    m.add(frustum(0.016, 0.016, 0.15, seg, (0, 0, 0.015)), "grip")
    m.add(sphere(0.028, max(seg // 2, 4), seg, (0, 0, -0.08)), "pommel")
    return m


def _table(seg: int) -> Mesh:
    m = Mesh().add(box((1.2, 0.7, 0.05), (0, 0, 0.725)), "top")
    for x in (-0.55, 0.55):
        for y in (-0.3, 0.3):
            m.add(box((0.05, 0.05, 0.7), (x, y, 0.35)), "leg")
    return m


def _chair(seg: int) -> Mesh:
    m = Mesh().add(box((0.45, 0.45, 0.05), (0, 0, 0.45)), "seat")
    m.add(box((0.45, 0.04, 0.45), (0, 0.205, 0.7)), "back")
    for x in (-0.2, 0.2):
        for y in (-0.2, 0.2):
            m.add(box((0.04, 0.04, 0.43), (x, y, 0.215)), "leg")
    return m


def _tree(seg: int) -> Mesh:
    m = Mesh().add(frustum(0.12, 0.09, 1.0, seg, (0, 0, 0.5)), "trunk")
    m.add(frustum(0.7, 0.0, 1.2, seg, (0, 0, 1.4)), "foliage")
    m.add(frustum(0.5, 0.0, 0.9, seg, (0, 0, 2.0)), "foliage_top")
    return m


def _cup(seg: int) -> Mesh:
    m = Mesh().add(frustum(0.04, 0.05, 0.1, seg, (0, 0, 0.05)), "body")
    m.add(box((0.015, 0.02, 0.06), (0.06, 0, 0.055)), "handle")
    return m


def _house(seg: int) -> Mesh:
    m = Mesh().add(box((4, 3, 2.5), (0, 0, 1.25)), "walls")
    m.add(frustum(2.9, 0.0, 1.6, 4, (0, 0, 3.3)), "roof")
    m.add(box((0.9, 0.1, 1.9), (0, -1.52, 0.95)), "door")
    return m


def _rocket(seg: int) -> Mesh:
    m = Mesh().add(frustum(0.3, 0.3, 2.0, seg, (0, 0, 1.2)), "body")
    m.add(frustum(0.3, 0.0, 0.7, seg, (0, 0, 2.55)), "nose")
    for x, y in ((0.35, 0), (-0.35, 0), (0, 0.35), (0, -0.35)):
        m.add(box((0.3 if x else 0.04, 0.3 if y else 0.04, 0.5), (x, y, 0.45)), "fin")
    return m


RECIPES: list[tuple[tuple[str, ...], str, Callable[[int], Mesh]]] = [
    (("sword", "blade", "검", "칼", "도검"), "sword", _sword),
    (("table", "desk", "테이블", "책상", "식탁"), "table", _table),
    (("chair", "stool", "의자"), "chair", _chair),
    (("tree", "pine", "나무", "소나무"), "tree", _tree),
    (("cup", "mug", "컵", "머그"), "cup", _cup),
    (("house", "home", "building", "집", "주택", "건물"), "house", _house),
    (("rocket", "로켓"), "rocket", _rocket),
    (("sphere", "ball", "orb", "구체"), "sphere", lambda seg: Mesh().add(sphere(0.5, seg // 2, seg), "sphere")),
    (("cylinder", "pillar", "column", "원기둥", "기둥"), "cylinder",
     lambda seg: Mesh().add(frustum(0.3, 0.3, 1.0, seg, (0, 0, 0.5)), "cylinder")),
    (("cone", "원뿔"), "cone", lambda seg: Mesh().add(frustum(0.4, 0.0, 1.0, seg, (0, 0, 0.5)), "cone")),
    (("cube", "box", "crate", "큐브", "상자", "박스"), "cube", lambda seg: Mesh().add(box((1, 1, 1), (0, 0, 0.5)), "cube")),
]


# Words that contain a recipe keyword but mean something else ("검은" = black).
_FALSE_FRIENDS = ("검은", "검정", "검색", "검사", "칼라")


def pick_recipe(prompt: str) -> tuple[str, Callable[[int], Mesh]]:
    text = prompt.lower()
    for word in _FALSE_FRIENDS:
        text = text.replace(word, " ")
    for words, name, build in RECIPES:
        if any(w in text for w in words):
            return name, build
    return "cube", RECIPES[-1][2]


def _segments_for_budget(max_polygons: int | None) -> int:
    if not max_polygons:
        return 32
    # Recipes have up to ~8 round parts at ~seg^2/2 faces; keep well under budget.
    return max(6, min(32, int(math.sqrt(max_polygons / 8))))


def generate_mesh(prompt: str, max_polygons: int | None = None) -> tuple[str, Mesh]:
    name, build = pick_recipe(prompt)
    return name, build(_segments_for_budget(max_polygons))


class ProceduralAdapter(GenerationAdapter):
    key = "procedural"

    async def generate(self, task: TaskSpec, model: ModelSpec, workdir: Path) -> GenerationResult:
        name, mesh = generate_mesh(task.prompt, task.max_polygons)
        # Emit a requested format directly when possible so no conversion is needed.
        fmt = next((f for f in task.output_formats if f in (OutputFormat.GLB, OutputFormat.OBJ)),
                   OutputFormat.OBJ)
        path = workdir / f"{name}.{fmt.value}"
        if fmt is OutputFormat.GLB:
            mesh.write_glb(path)
        else:
            mesh.write_obj(path)
        return GenerationResult(
            path=path, format=fmt,
            metadata={"recipe": name, "parts": mesh.parts, "faces": len(mesh.faces)},
        )
