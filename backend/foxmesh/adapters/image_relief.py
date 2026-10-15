"""Built-in image-to-3D on CPU: turns an image's silhouette into a closed, textured 2.5D mesh.

Steps: remove the background (alpha channel, or colour distance from the border),
clean the mask, then build a grid mesh whose front and back surfaces follow one of
three shapes:

* ``inflate`` (default) - a rounded, pillow-like body from the distance to the outline
* ``relief`` - flat back with a height map from image brightness (lithophane-like)
* ``extrude`` - flat slab with straight walls (logos, signs, keychains)

The image is mapped onto the front and back as a texture. This is not a true
3D reconstruction; TripoSR, TRELLIS etc. do that when an inference server is
connected, and the router prefers them when they are available.
"""

from __future__ import annotations

import asyncio
import io
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..schemas import ModelSpec, OutputFormat, TaskSpec
from .base import GenerationAdapter, GenerationResult
from .mesh import Mesh

ANALYSIS_SIZE = 256
DEFAULT_GRID = 128
MIN_GRID, MAX_GRID = 16, 200
TEXTURE_SIZE = 1024

MODE_WORDS = {
    "relief": ("부조", "릴리프", "높이맵", "리토판", "relief", "heightmap", "height map", "lithophane"),
    "extrude": ("평면", "납작", "쿠키", "로고", "키링", "간판", "돌출", "flat", "extrude", "logo",
                "keychain", "sign", "cookie"),
    "inflate": ("부풀", "통통", "인형", "봉제", "inflate", "puffy", "plush", "balloon"),
}


@dataclass
class ImageMeshInfo:
    mode: str
    mask_method: str
    grid: tuple[int, int]  # (columns, rows)
    coverage: float


def pick_mode(prompt: str) -> str | None:
    text = prompt.lower()
    for mode, words in MODE_WORDS.items():
        if any(w in text for w in words):
            return mode
    return None


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    total = weight[-1]
    mean = np.cumsum(hist * centers)
    w0, w1 = weight[:-1], total - weight[:-1]
    valid = (w0 > 0) & (w1 > 0)
    m0 = np.where(valid, mean[:-1] / np.maximum(w0, 1), 0)
    m1 = np.where(valid, (mean[-1] - mean[:-1]) / np.maximum(w1, 1), 0)
    between = np.where(valid, w0 * w1 * (m0 - m1) ** 2, 0)
    return float(centers[int(np.argmax(between))])


def _border(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]])


def foreground_mask(rgba: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (mask, method). ``method`` is "alpha", "background" or "full"."""
    alpha = rgba[..., 3]
    if (alpha < 250).mean() > 0.01:
        mask = alpha > 127
        method = "alpha"
    else:
        rgb = rgba[..., :3].astype(np.float32)
        border = _border(rgb)
        if border.std(axis=0).max() > 40:
            # No uniform background to separate from: use the whole image.
            return np.ones(alpha.shape, bool), "full"
        background = np.median(border, axis=0)
        dist = np.linalg.norm(rgb - background, axis=-1)
        threshold = max(30.0, _otsu(dist))
        mask = dist > threshold
        method = "background"
    coverage = mask.mean()
    if coverage < 0.01 or coverage > 0.97:
        return np.ones(alpha.shape, bool), "full"
    return mask, method


def _flood(mask: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    """4-connected flood fill over True cells of ``mask`` starting from ``seeds``."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    queue = deque((r, c) for r, c in seeds if mask[r, c])
    for r, c in queue:
        seen[r, c] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def largest_component(mask: np.ndarray) -> np.ndarray:
    remaining = mask.copy()
    best = np.zeros_like(mask)
    while remaining.any():
        r, c = map(int, np.argwhere(remaining)[0])
        component = _flood(remaining, [(r, c)])
        if component.sum() > best.sum():
            best = component
        remaining &= ~component
    return best


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions not connected to the image border (PIL flood fill, in C)."""
    # copy(): an image made by fromarray can share the read-only array buffer.
    padded = Image.fromarray(np.pad(mask, 1).astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(padded, (0, 0), 128)
    return (np.asarray(padded) != 128)[1:-1, 1:-1]


def remove_diagonal_contacts(mask: np.ndarray) -> np.ndarray:
    """Fill cells so no two filled cells touch only at a corner (that would be non-manifold)."""
    mask = mask.copy()
    while True:
        a, b = mask[:-1, :-1], mask[:-1, 1:]
        c, d = mask[1:, :-1], mask[1:, 1:]
        diag1 = a & d & ~b & ~c  # filled TL and BR only
        diag2 = b & c & ~a & ~d  # filled TR and BL only
        if not (diag1.any() or diag2.any()):
            return mask
        mask[:-1, 1:] |= diag1  # fill TR
        mask[:-1, :-1] |= diag2  # fill TL


def clean_mask(mask: np.ndarray) -> np.ndarray:
    return remove_diagonal_contacts(fill_holes(largest_component(mask)))


def distance_to_edge(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance (in cells) from each filled cell to the outside."""
    dist = np.zeros(mask.shape, np.float32)
    current = mask.copy()
    step = 0
    while current.any():
        step += 1
        dist[current] = step
        padded = np.pad(current, 1, constant_values=False)
        eroded = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        if step % 2 == 0:
            # Alternating 4- and 8-neighbour erosion approximates a round metric.
            eroded &= padded[:-2, :-2] & padded[:-2, 2:] & padded[2:, :-2] & padded[2:, 2:]
        current = eroded
    return dist


def bleed_colors(rgb: np.ndarray, mask: np.ndarray, passes: int = 24) -> np.ndarray:
    """Spread foreground colours into the background.

    Outline vertices sample the texture right at the silhouette; without this the
    side walls pick up background (white, or black from transparent pixels).
    """
    if not mask.any():
        return rgb
    colors = np.where(mask[..., None], rgb, 0).astype(np.float32)
    known = mask.copy()
    for _ in range(passes):
        if known.all():
            break
        p_col = np.pad(colors, ((1, 1), (1, 1), (0, 0)))
        p_known = np.pad(known, 1).astype(np.float32)
        total = np.zeros_like(colors)
        count = np.zeros(known.shape, np.float32)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sl = (slice(1 + dr, p_known.shape[0] - 1 + dr), slice(1 + dc, p_known.shape[1] - 1 + dc))
            total += p_col[sl] * p_known[sl][..., None]
            count += p_known[sl]
        grow = ~known & (count > 0)
        colors[grow] = total[grow] / count[grow][:, None]
        known |= grow
    colors[~known] = rgb[mask].mean(axis=0)
    return colors.clip(0, 255).astype(np.uint8)


def _blur(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(arr, 1, mode="edge")
        arr = sum(p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
                  for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9.0
    return arr


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def grid_size(aspect: float, coverage: float, max_polygons: int | None) -> int:
    """Long-side cell count. Each filled cell costs ~4 triangles (front + back)."""
    if not max_polygons:
        return DEFAULT_GRID
    short_ratio = min(aspect, 1 / aspect)
    cells = max_polygons * 0.8 / 4
    n = int((cells / max(coverage * short_ratio, 1e-3)) ** 0.5)
    return max(MIN_GRID, min(MAX_GRID, n))


def build_mesh(mask: np.ndarray, front: np.ndarray, back: np.ndarray, cell: float) -> Mesh:
    """Closed mesh over the filled cells. ``front``/``back`` are per-cell offsets (m) from z=0.

    Built Z-up: image x -> +X, image up -> +Z, the front faces -Y (Blender's front view).
    """
    h, w = mask.shape
    mesh = Mesh(uvs=[], smooth=True)
    front_idx = -np.ones((h + 1, w + 1), np.int64)
    back_idx = -np.ones((h + 1, w + 1), np.int64)

    # Corner offsets: mean over the filled cells around each corner.
    padded = np.pad(mask, 1).astype(np.float32)
    count = padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]

    def corner(values: np.ndarray) -> np.ndarray:
        p = np.pad(np.where(mask, values, 0.0), 1)
        total = p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]
        return np.divide(total, count, out=np.zeros_like(total), where=count > 0)

    corner_front, corner_back = corner(front), corner(back)
    for r, c in np.argwhere(count > 0):
        x, z = (c - w / 2) * cell, (h / 2 - r) * cell
        uv = (c / w, r / h)
        front_idx[r, c] = len(mesh.vertices)
        mesh.vertices.append((x, -float(corner_front[r, c]), z))
        mesh.uvs.append(uv)
        back_idx[r, c] = len(mesh.vertices)
        mesh.vertices.append((x, float(corner_back[r, c]), z))
        mesh.uvs.append(uv)

    def filled(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and bool(mask[r, c])

    outline: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def wall(a: tuple[int, int], b_: tuple[int, int]) -> None:
        mesh.faces.append((f[a], f[b_], b[b_], b[a]))
        outline.setdefault(a, set()).add(b_)
        outline.setdefault(b_, set()).add(a)

    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        tl, tr, br, bl = (r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)
        f = {k: int(front_idx[k]) for k in (tl, tr, br, bl)}
        b = {k: int(back_idx[k]) for k in (tl, tr, br, bl)}
        mesh.faces.append((f[bl], f[br], f[tr], f[tl]))  # front, normal -Y
        mesh.faces.append((b[tl], b[tr], b[br], b[bl]))  # back, normal +Y
        # Side walls where the neighbouring cell is empty (outward-facing).
        if not filled(r - 1, c):
            wall(tl, tr)
        if not filled(r + 1, c):
            wall(br, bl)
        if not filled(r, c - 1):
            wall(bl, tl)
        if not filled(r, c + 1):
            wall(tr, br)

    _smooth_outline(mesh, outline, front_idx, back_idx, cell, w, h)
    mesh.parts.append("image")
    return mesh


def _smooth_outline(mesh: Mesh, outline: dict, front_idx: np.ndarray, back_idx: np.ndarray,
                    cell: float, w: int, h: int, iterations: int = 4) -> None:
    """Relax the outline corners along the outline so pixel stair-steps become smooth curves.

    Only the in-plane position moves (front and back corners together); UVs follow so
    the texture stays aligned. Corners with more than two outline neighbours are kept.
    """
    pos = {k: np.array(k, np.float64) for k in outline}  # (row, col) as floats
    for _ in range(iterations):
        nxt = {}
        for k, neighbours in outline.items():
            if len(neighbours) == 2:
                n1, n2 = neighbours
                nxt[k] = 0.5 * pos[k] + 0.25 * (pos[n1] + pos[n2])
            else:
                nxt[k] = pos[k]
        pos = nxt
    for k, (r, c) in pos.items():
        x, z = (c - w / 2) * cell, (h / 2 - r) * cell
        for idx in (int(front_idx[k]), int(back_idx[k])):
            mesh.vertices[idx] = (x, mesh.vertices[idx][1], z)
            mesh.uvs[idx] = (c / w, r / h)


def image_to_mesh(
    image_path: Path,
    prompt: str = "",
    max_polygons: int | None = None,
    size_m: float | None = None,
) -> tuple[Mesh, ImageMeshInfo]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGBA")

    # 1) Foreground mask on a small copy, then crop the image to the object.
    small = image.copy()
    small.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    mask_small, method = foreground_mask(np.asarray(small))
    mask_small = clean_mask(mask_small) if method != "full" else mask_small
    rows, cols = np.nonzero(mask_small)
    sx, sy = image.width / small.width, image.height / small.height
    margin = 1
    box = (
        max(int((cols.min() - margin) * sx), 0), max(int((rows.min() - margin) * sy), 0),
        min(int((cols.max() + 1 + margin) * sx), image.width),
        min(int((rows.max() + 1 + margin) * sy), image.height),
    )
    cropped = image.crop(box)
    mask_crop = Image.fromarray((mask_small * 255).astype(np.uint8)).resize(
        image.size, Image.NEAREST).crop(box)

    # 2) Grid resolution from the polygon budget.
    coverage = float(np.asarray(mask_crop).mean() / 255)
    aspect = cropped.width / cropped.height
    n = grid_size(aspect, coverage, max_polygons)
    cols_n, rows_n = (n, max(2, round(n / aspect))) if aspect >= 1 else (max(2, round(n * aspect)), n)
    mask = np.asarray(mask_crop.resize((cols_n, rows_n), Image.BILINEAR)) > 127
    if method != "full":
        mask = clean_mask(mask)
    if not mask.any():
        mask = np.ones((rows_n, cols_n), bool)
        method = "full"

    # 3) Shape of the front/back surfaces.
    mode = pick_mode(prompt) or ("relief" if method == "full" else "inflate")
    width_m = size_m or 1.0
    cell = width_m / max(cols_n, rows_n)
    if mode == "inflate":
        dist = distance_to_edge(mask)
        t = dist / max(dist.max(), 1.0)
        profile = np.sqrt(np.clip(2 * t - t * t, 0, 1))  # circular cross-section
        depth = 0.01 * width_m + 0.2 * width_m * min(1.0, dist.max() / (0.25 * max(cols_n, rows_n))) * profile
        front = back = depth
    elif mode == "relief":
        gray = np.asarray(cropped.convert("L").resize((cols_n, rows_n), Image.BILINEAR), np.float32) / 255
        front = 0.01 * width_m + 0.05 * width_m * _blur(gray)
        back = np.full(mask.shape, 0.01 * width_m, np.float32)
    else:
        front = back = np.full(mask.shape, 0.04 * width_m, np.float32)

    mesh = build_mesh(mask, front, back, cell)
    texture = cropped.copy()
    texture.thumbnail((TEXTURE_SIZE, TEXTURE_SIZE))
    if method != "full":
        # Re-detect the foreground at texture resolution: the upscaled analysis mask is
        # too coarse and would let background pixels count as object colour.
        tex_rgba = np.asarray(texture.convert("RGBA"))
        tex_mask, tex_method = foreground_mask(tex_rgba)
        if tex_method == "full":
            tex_mask = np.asarray(mask_crop.resize(texture.size, Image.BILINEAR)) > 127
        tex_mask = fill_holes(tex_mask)
        texture = Image.fromarray(bleed_colors(tex_rgba[..., :3], tex_mask))
    else:
        texture = texture.convert("RGB")
    buf = io.BytesIO()
    texture.save(buf, format="PNG", optimize=True)
    mesh.texture_png = buf.getvalue()
    return mesh, ImageMeshInfo(mode=mode, mask_method=method, grid=(cols_n, rows_n),
                               coverage=round(float(mask.mean()), 3))


class ImageReliefAdapter(GenerationAdapter):
    key = "image_relief"

    async def generate(self, task: TaskSpec, model: ModelSpec, workdir: Path) -> GenerationResult:
        if not task.image_path:
            raise ValueError("image_to_3d needs an input image")
        mesh, info = await asyncio.to_thread(
            image_to_mesh, Path(task.image_path), task.prompt, task.max_polygons, task.target_size_m
        )
        fmt = next((f for f in task.output_formats if f in (OutputFormat.GLB, OutputFormat.OBJ)),
                   OutputFormat.GLB)
        path = workdir / f"image_model.{fmt.value}"
        await asyncio.to_thread(mesh.write_glb if fmt is OutputFormat.GLB else mesh.write_obj, path)
        return GenerationResult(path=path, format=fmt, metadata={
            "mode": info.mode, "mask": info.mask_method, "grid": list(info.grid),
            "coverage": info.coverage, "faces": len(mesh.faces),
        })
