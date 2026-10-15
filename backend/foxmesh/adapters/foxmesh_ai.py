"""Adapter for FoxMesh's own model (the ``foxmesh-ai`` package).

The model descriptor is produced by ``foxmesh-ai models export --to $FOXMESH_MODELS_DIR``
and points at a model store and version. The loaded model is kept in memory and
reused across jobs; it is reloaded when the store's version changes.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from ..schemas import ModelSpec, OutputFormat, TaskSpec
from .base import GenerationAdapter, GenerationResult


class FoxMeshAIAdapter(GenerationAdapter):
    key = "foxmesh_ai"

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[float, object]] = {}
        self._lock = threading.Lock()

    def _checkpoint(self, model: ModelSpec) -> Path:
        from foxmesh_ai.store import ModelStore

        store = model.config.get("store")
        if not store:
            raise LookupError("config.store is not set")
        return ModelStore(store).checkpoint(model.config.get("version", "production"))

    def unavailable_reason(self, model: ModelSpec) -> str | None:
        try:
            import foxmesh_ai.generate  # noqa: F401  (imports torch)
        except ImportError:
            return "foxmesh-ai is not installed (pip install -e ai[torch])"
        try:
            path = self._checkpoint(model)
        except LookupError as exc:
            return str(exc)
        if not path.exists():
            return f"checkpoint missing: {path}"
        return None

    def _generator(self, model: ModelSpec):
        from foxmesh_ai.generate import Generator

        path = self._checkpoint(model)
        key = (str(path), model.config.get("device", "auto"))
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(key)
            if cached is None or cached[0] != mtime:
                self._cache[key] = (mtime, Generator(path, device=key[1]))
            return self._cache[key][1]

    def _run(self, task: TaskSpec, model: ModelSpec, path: Path) -> dict:
        from foxmesh_ai.geometry import write_mesh

        generator = self._generator(model)
        resolution = int(model.config.get("resolution", 64))
        if task.max_polygons:
            # About 2 * resolution^2 * coverage triangles; keep a coarse grid for small budgets.
            resolution = max(32, min(resolution, int((task.max_polygons / 4) ** 0.5)))
        mesh = generator.generate(task.prompt or None, task.image_path, resolution=resolution,
                                  size_m=task.target_size_m or 1.0)
        write_mesh(mesh, path)
        return {"resolution": resolution, "faces": int(len(mesh.faces)), "seconds": mesh.meta["seconds"],
                "version": model.config.get("version")}

    async def generate(self, task: TaskSpec, model: ModelSpec, workdir: Path) -> GenerationResult:
        fmt = next((f for f in task.output_formats if f in (OutputFormat.GLB, OutputFormat.OBJ)),
                   OutputFormat.GLB)
        path = workdir / f"foxmesh.{fmt.value}"
        meta = await asyncio.to_thread(self._run, task, model, path)
        return GenerationResult(path=path, format=fmt, metadata=meta)
