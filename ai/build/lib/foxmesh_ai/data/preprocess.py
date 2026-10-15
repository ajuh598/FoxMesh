"""Turn dataset meshes into training samples, cached as ``cache/<item id>.npz``.

Per item:

* ``uniform_points`` / ``uniform_occ``: points uniform in [-1, 1]^3 with inside labels
* ``near_points`` / ``near_occ``: surface samples jittered at two scales, where the
  decision boundary is learned
* ``surface_points`` / ``surface_colors``: exact surface samples for colour training
  and for Chamfer / F-score evaluation
* ``renders``: RGBA conditioning images from several viewpoints (image -> 3D)

Inside labels come from a 3-axis ray-parity voxelisation at ``grid`` resolution.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry import grid_lookup, load_mesh, sample_surface, voxelize
from ..render import DEFAULT_VIEWS, render
from .dataset import Dataset

CACHE_VERSION = 1


@dataclass
class PreprocessConfig:
    grid: int = 128
    uniform: int = 16384
    near: int = 16384
    surface: int = 8192
    near_sigmas: tuple[float, float] = (0.01, 0.05)
    render_size: int = 128
    views: tuple = tuple(DEFAULT_VIEWS)


def preprocess_mesh(mesh_path: Path, out_path: Path, cfg: PreprocessConfig, seed: int,
                    images: list[Path] = ()) -> None:
    rng = np.random.default_rng(seed)
    mesh = load_mesh(mesh_path)
    grid = voxelize(mesh, cfg.grid)

    uniform = rng.uniform(-1, 1, (cfg.uniform, 3)).astype(np.float32)
    surf, _, _ = sample_surface(mesh, cfg.near, rng)
    sigma = np.where(rng.random(cfg.near) < 0.5, *cfg.near_sigmas).astype(np.float32)
    near = np.clip(surf + rng.normal(size=surf.shape).astype(np.float32) * sigma[:, None], -1, 1)

    surface, _, colors = sample_surface(mesh, cfg.surface, rng)
    renders = [render(mesh, az, el, cfg.render_size, seed=i) for i, (az, el) in enumerate(cfg.views)]
    # Real reference images (if any) are used as extra conditioning views.
    for image in images:
        img = Image.open(image).convert("RGBA").resize((cfg.render_size, cfg.render_size))
        renders.append(np.asarray(img))

    np.savez_compressed(
        out_path,
        version=CACHE_VERSION,
        uniform_points=uniform.astype(np.float16), uniform_occ=grid_lookup(grid, uniform),
        near_points=near.astype(np.float16), near_occ=grid_lookup(grid, near),
        surface_points=surface.astype(np.float16),
        surface_colors=(np.zeros_like(surface) if colors is None else colors * 255).astype(np.uint8),
        has_colors=colors is not None,
        renders=np.stack(renders).astype(np.uint8),
    )


def _work(args) -> tuple[str, str | None]:
    item_id, mesh_path, out_path, cfg, seed, images = args
    try:
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        preprocess_mesh(mesh_path, tmp, cfg, seed, images)
        tmp.replace(out_path)
        return item_id, None
    except Exception as exc:  # report and continue with the rest
        return item_id, f"{type(exc).__name__}: {exc}"


def preprocess_dataset(dataset: Dataset, cfg: PreprocessConfig | None = None, workers: int | None = None,
                       force: bool = False, progress=None) -> dict:
    cfg = cfg or PreprocessConfig()
    cache = dataset.root / "cache"
    cache.mkdir(exist_ok=True)
    jobs = []
    for n, item in enumerate(dataset):
        out = cache / f"{item.id}.npz"
        if out.exists() and not force:
            continue
        jobs.append((item.id, dataset.path(item.mesh), out, cfg, n,
                     [dataset.path(i) for i in item.images]))
    errors: dict[str, str] = {}
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    if workers == 1 or len(jobs) < 4:
        results = map(_work, jobs)
    else:
        pool = ProcessPoolExecutor(workers)
        results = pool.map(_work, jobs, chunksize=4)
    for done, (item_id, error) in enumerate(results, 1):
        if error:
            errors[item_id] = error
        if progress:
            progress(done, len(jobs))
    if workers != 1 and len(jobs) >= 4:
        pool.shutdown()
    return {"processed": len(jobs) - len(errors), "skipped": len(dataset) - len(jobs), "errors": errors}


def load_sample(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
