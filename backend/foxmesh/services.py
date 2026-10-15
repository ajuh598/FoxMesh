"""Wire the components together from :class:`~foxmesh.config.Settings`."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import AdapterRegistry
from .blender import BlenderWorker, LitePostProcessor
from .config import Settings
from .gpu import GpuMonitor
from .jobs import JobManager
from .pipeline import Pipeline
from .registry import ModelRegistry, StatsStore
from .router import PolicyStore, Router


@dataclass
class Services:
    settings: Settings
    adapters: AdapterRegistry
    registry: ModelRegistry
    gpu: GpuMonitor
    policies: PolicyStore
    router: Router
    pipeline: Pipeline
    jobs: JobManager


def build_services(
    settings: Settings | None = None,
    adapters: AdapterRegistry | None = None,
    gpu: GpuMonitor | None = None,
) -> Services:
    settings = settings or Settings()
    settings.ensure_dirs()
    adapters = adapters or AdapterRegistry()
    stats = StatsStore(settings.stats_path, settings.run_log_path)
    registry = ModelRegistry(stats=stats, availability=adapters.unavailable_reason)
    registry.load_builtin()
    if settings.models_dir:
        registry.load_dir(settings.models_dir)
    gpu = gpu or GpuMonitor(vram_override_gb=settings.vram_override_gb)
    policies = PolicyStore(settings.policies_path)
    router = Router(registry, gpu, policies, blender_available=settings.blender_available)
    if settings.blender_available:
        postprocessor = BlenderWorker(
            settings.blender_executable, mode=settings.blender_mode,
            timeout_s=settings.blender_timeout_s, thumbnails=settings.thumbnails,
        )
    else:
        postprocessor = LitePostProcessor()
    pipeline = Pipeline(
        registry, adapters, router, gpu, postprocessor, settings.jobs_dir,
        max_attempts=settings.max_attempts, transient_retries=settings.transient_retries,
        generation_timeout_s=settings.generation_timeout_s,
    )
    jobs = JobManager(
        pipeline, settings.uploads_dir, concurrency=settings.worker_concurrency,
        dedup_ttl_s=settings.dedup_ttl_s, commercial_use=settings.commercial_use,
    )
    return Services(settings, adapters, registry, gpu, policies, router, pipeline, jobs)
