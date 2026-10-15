"""Fast conditioning-image renderer: shaded point splatting with a depth test, in NumPy.

Used to make the image half of the training pairs (image -> 3D). It renders a
mesh from a given azimuth/elevation with an orthographic camera onto a
transparent background, similar to product shots with the background removed.
"""

from __future__ import annotations

import numpy as np

from .geometry import TriMesh, sample_surface

DEFAULT_VIEWS = [(0.0, 15.0), (35.0, 20.0), (-35.0, 20.0), (90.0, 10.0)]


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rx @ ry


def render(mesh: TriMesh, azimuth: float = 0.0, elevation: float = 15.0, size: int = 128,
           points: int | None = None, seed: int = 0) -> np.ndarray:
    """Return an RGBA uint8 image (size, size, 4). The camera looks down -Z (glTF front)."""
    rng = np.random.default_rng(seed)
    n_points = points or size * size * 6
    pts, normals, colors = sample_surface(mesh, n_points, rng)
    if colors is None:
        colors = np.full_like(pts, 0.7)
    rot = _rotation(azimuth, elevation)
    p = pts @ rot.T
    n = normals @ rot.T
    light = np.array([0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)  # two-sided: sampling does not know inside/outside
    rgb = np.clip(colors * shade[:, None], 0, 1)

    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    scale = 0.9 * size / max(float((hi - lo).max()), 1e-6)
    center = (lo + hi) / 2
    xy = (p[:, :2] - center) * scale + size / 2
    px = np.clip(xy[:, 0].astype(int), 0, size - 1)
    py = np.clip((size - 1 - xy[:, 1]).astype(int), 0, size - 1)

    order = np.argsort(p[:, 2])  # far (small z) first; nearer points overwrite
    img = np.zeros((size, size, 4), np.float32)
    for dx in (0, 1):  # 2x2 splats close the gaps between samples
        for dy in (0, 1):
            x = np.clip(px[order] + dx, 0, size - 1)
            y = np.clip(py[order] + dy, 0, size - 1)
            img[y, x, :3] = rgb[order]
            img[y, x, 3] = 1.0
    return (img * 255).astype(np.uint8)


def render_views(mesh: TriMesh, views=DEFAULT_VIEWS, size: int = 128) -> list[np.ndarray]:
    return [render(mesh, az, el, size, seed=i) for i, (az, el) in enumerate(views)]
