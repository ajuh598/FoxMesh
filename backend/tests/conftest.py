from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path

import pytest

from foxmesh.gpu import GpuMonitor
from foxmesh.registry import ModelRegistry, StatsStore
from foxmesh.schemas import ModelSpec, QualityPriors

# Path to a Blender executable or a Python with ``bpy`` for the real-Blender tests.
BLENDER = os.environ.get("FOXMESH_TEST_BLENDER")
BLENDER_MODE = os.environ.get("FOXMESH_TEST_BLENDER_MODE", "bpy")


def make_model(model_id: str, **overrides) -> ModelSpec:
    data = {
        "id": model_id, "name": model_id, "version": "1", "adapter": "fake",
        "tasks": ["generate_3d"], "inputs": ["text"], "outputs": ["glb"],
        "runtime": "local_gpu", "vram_gb": 8, "est_runtime_s": 30,
        "quality": QualityPriors(accuracy=0.6, mesh=0.6, texture=0.6),
        "license": "MIT", "commercial_use": True,
    }
    data.update(overrides)
    return ModelSpec.model_validate(data)


def fixed_gpu(total: float = 24.0, free: float | None = None) -> GpuMonitor:
    free = total if free is None else free
    return GpuMonitor(probe=lambda: [{"index": 0, "name": "test", "total_gb": total,
                                      "used_gb": total - free, "free_gb": free,
                                      "utilization": 0}])


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(StatsStore())


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    return tmp_path


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run ``async def`` tests on a fresh event loop (no pytest-asyncio dependency)."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**args))
        return True
    return None
