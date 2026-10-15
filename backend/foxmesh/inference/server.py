"""Inference server that the ``http`` adapter talks to.

One server process owns a GPU. Runners are loaded lazily on first use and kept
in memory so later requests skip model loading; idle runners are unloaded when
another runner needs the VRAM. Requests to the same runner are serialised.

Run with::

    uvicorn foxmesh.inference.server:app --port 9000
    FOXMESH_INFERENCE_RUNNERS=mypkg.runners:TripoSRRunner,mypkg.runners:TrellisRunner
    FOXMESH_INFERENCE_VRAM_GB=24        # evict least recently used runners beyond this
    FOXMESH_INFERENCE_PRELOAD=triposr   # load at startup instead of on first request
    FOXMESH_INFERENCE_IDLE_S=1800       # unload runners idle this long

A runner wraps one PyTorch model::

    class TripoSRRunner(ModelRunner):
        name = "triposr"
        vram_gb = 6

        def load(self):
            from tsr.system import TSR
            self.model = TSR.from_pretrained("stabilityai/TripoSR", ...).to("cuda")

        def generate(self, request, out_path):
            ...  # run the model, write the mesh to out_path
            return out_path
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    model: str
    task_type: str = "generate_3d"
    prompt: str = ""
    image_b64: str | None = None
    output_format: str = "glb"
    params: dict[str, Any] = Field(default_factory=dict)

    def image_bytes(self) -> bytes | None:
        return base64.b64decode(self.image_b64) if self.image_b64 else None


class ModelRunner(ABC):
    name: str
    vram_gb: float = 0.0
    output_formats: tuple[str, ...] = ("glb",)

    def __init__(self) -> None:
        self.loaded = False
        self.last_used = 0.0
        self.lock = asyncio.Lock()

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device (called once, in a worker thread)."""

    def unload(self) -> None:
        """Free device memory. Override to drop references and empty the CUDA cache."""
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @abstractmethod
    def generate(self, request: InferenceRequest, out_path: Path) -> Path:
        """Run inference and write the mesh to ``out_path`` (blocking)."""


class ProceduralRunner(ModelRunner):
    """CPU stand-in used for tests and for checking a deployment end to end."""

    name = "procedural"
    output_formats = ("obj",)

    def load(self) -> None:
        pass

    def generate(self, request: InferenceRequest, out_path: Path) -> Path:
        from ..adapters.procedural import generate_mesh

        _, mesh = generate_mesh(request.prompt, request.params.get("max_faces"))
        path = out_path.with_suffix(".obj")
        mesh.write_obj(path)
        return path


class RunnerPool:
    def __init__(self, runners: list[ModelRunner], vram_budget_gb: float | None = None):
        self.runners = {r.name: r for r in runners}
        self.vram_budget_gb = vram_budget_gb
        self._load_lock = asyncio.Lock()

    def _loaded_vram(self) -> float:
        return sum(r.vram_gb for r in self.runners.values() if r.loaded)

    async def acquire(self, name: str) -> ModelRunner:
        runner = self.runners.get(name)
        if runner is None:
            raise KeyError(name)
        async with self._load_lock:
            if not runner.loaded:
                if self.vram_budget_gb is not None:
                    # Evict least recently used idle runners until the new one fits.
                    idle = sorted((r for r in self.runners.values()
                                   if r.loaded and not r.lock.locked()), key=lambda r: r.last_used)
                    while idle and self._loaded_vram() + runner.vram_gb > self.vram_budget_gb:
                        victim = idle.pop(0)
                        log.info("unloading %s to free VRAM", victim.name)
                        await asyncio.to_thread(victim.unload)
                        victim.loaded = False
                t0 = time.monotonic()
                await asyncio.to_thread(runner.load)
                runner.loaded = True
                runner.last_used = time.time()
                log.info("loaded %s in %.1fs", name, time.monotonic() - t0)
        return runner


    async def preload(self, names: list[str]) -> None:
        """Load runners up front so the first request does not pay the loading time."""
        for name in names:
            await self.acquire(name)

    async def unload_idle(self, idle_s: float, now: float | None = None) -> list[str]:
        """Unload runners unused for ``idle_s`` seconds, freeing VRAM for other work."""
        now = now or time.time()
        unloaded = []
        async with self._load_lock:
            for runner in self.runners.values():
                if runner.loaded and not runner.lock.locked() and now - runner.last_used >= idle_s:
                    await asyncio.to_thread(runner.unload)
                    runner.loaded = False
                    unloaded.append(runner.name)
        return unloaded


def _load_runner_classes(spec: str) -> list[ModelRunner]:
    runners: list[ModelRunner] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        module, _, attr = item.partition(":")
        runners.append(getattr(importlib.import_module(module), attr)())
    return runners


def create_inference_app(pool: RunnerPool | None = None) -> FastAPI:
    if pool is None:
        runners: list[ModelRunner] = [ProceduralRunner()]
        if spec := os.environ.get("FOXMESH_INFERENCE_RUNNERS"):
            runners += _load_runner_classes(spec)
        budget = os.environ.get("FOXMESH_INFERENCE_VRAM_GB")
        pool = RunnerPool(runners, float(budget) if budget else None)

    preload = [n.strip() for n in os.environ.get("FOXMESH_INFERENCE_PRELOAD", "").split(",") if n.strip()]
    idle_s = float(os.environ.get("FOXMESH_INFERENCE_IDLE_S", "0"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await pool.preload(preload)
        reaper = None
        if idle_s > 0:
            async def reap() -> None:
                while True:
                    await asyncio.sleep(min(idle_s, 60))
                    for name in await pool.unload_idle(idle_s):
                        log.info("unloaded idle runner %s", name)
            reaper = asyncio.create_task(reap())
        yield
        if reaper:
            reaper.cancel()

    app = FastAPI(title="FoxMesh inference", lifespan=lifespan)
    app.state.pool = pool

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"runners": {n: {"loaded": r.loaded, "vram_gb": r.vram_gb, "busy": r.lock.locked()}
                            for n, r in pool.runners.items()}}

    @app.post("/generate")
    async def generate(request: InferenceRequest) -> Response:
        try:
            runner = await pool.acquire(request.model)
        except KeyError as exc:
            raise HTTPException(404, f"unknown model {request.model!r}") from exc
        async with runner.lock:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"result.{request.output_format}"
                try:
                    path = await asyncio.to_thread(runner.generate, request, out)
                except Exception as exc:
                    log.exception("inference failed")
                    raise HTTPException(500, f"inference failed: {exc}") from exc
                finally:
                    runner.last_used = time.time()
                data = path.read_bytes()
        return Response(data, media_type="application/octet-stream",
                        headers={"X-FoxMesh-Format": path.suffix.lstrip(".")})

    return app


app = create_inference_app()
