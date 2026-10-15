"""GPU / VRAM monitoring and reservation.

The router reads a snapshot to drop models that cannot fit; the executor
reserves VRAM for the duration of a local inference so concurrent jobs never
oversubscribe the card.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import time

from .schemas import GpuSnapshot

_CACHE_TTL_S = 2.0


def query_nvidia_smi() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            devices.append({
                "index": int(parts[0]),
                "name": parts[1],
                "total_gb": float(parts[2]) / 1024,
                "used_gb": float(parts[3]) / 1024,
                "free_gb": float(parts[4]) / 1024,
                "utilization": float(parts[5]) if parts[5].replace(".", "").isdigit() else None,
            })
        except ValueError:
            continue
    return devices


class GpuMonitor:
    def __init__(self, vram_override_gb: float | None = None, probe=query_nvidia_smi):
        self._override = vram_override_gb
        self._probe = probe
        self._cached: tuple[float, list[dict] | None] | None = None
        self._reserved = 0.0
        self._cond = asyncio.Condition()

    def _devices(self) -> list[dict] | None:
        now = time.monotonic()
        if self._cached and now - self._cached[0] < _CACHE_TTL_S:
            return self._cached[1]
        devices = self._probe() if self._probe else None
        self._cached = (now, devices)
        return devices

    def snapshot(self) -> GpuSnapshot:
        devices = self._devices()
        if devices:
            # Models run on a single device, so the largest free device is what counts.
            total = max(d["total_gb"] for d in devices)
            free = max(d["free_gb"] for d in devices)
            detected = True
        else:
            total = free = self._override or 0.0
            detected = False
            devices = []
        # Live free memory already reflects models that have finished loading, while
        # total-minus-reserved covers jobs that are still loading; the smaller one is safe
        # and never counts the same job twice.
        schedulable = min(free, total - self._reserved)
        return GpuSnapshot(
            detected=detected,
            total_vram_gb=round(total, 2),
            free_vram_gb=round(max(schedulable, 0.0), 2),
            reserved_vram_gb=round(self._reserved, 2),
            devices=devices,
        )

    @property
    def capacity_gb(self) -> float:
        snap = self.snapshot()
        return snap.total_vram_gb

    @contextlib.asynccontextmanager
    async def reserve(self, vram_gb: float, timeout_s: float | None = None):
        """Hold ``vram_gb`` of VRAM, waiting until enough is free."""
        if vram_gb <= 0:
            yield
            return
        if vram_gb > self.capacity_gb:
            raise RuntimeError(
                f"model needs {vram_gb:g} GB VRAM but the GPU has {self.capacity_gb:g} GB"
            )
        async with self._cond:
            await asyncio.wait_for(
                self._cond.wait_for(lambda: self.snapshot().free_vram_gb >= vram_gb),
                timeout_s,
            )
            self._reserved += vram_gb
        try:
            yield
        finally:
            async with self._cond:
                self._reserved -= vram_gb
                self._cached = None
                self._cond.notify_all()
