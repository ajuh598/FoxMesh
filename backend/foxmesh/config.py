"""Runtime settings, read from environment variables (prefix ``FOXMESH_``)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"FOXMESH_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./var")).resolve())
    # Extra directory of model descriptor JSON files merged into the registry.
    models_dir: Path | None = field(
        default_factory=lambda: Path(p).resolve() if (p := _env("MODELS_DIR")) else None
    )
    # Path to a Blender executable, or to a Python interpreter with the ``bpy`` module.
    blender_executable: str | None = field(
        default_factory=lambda: _env("BLENDER") or shutil.which("blender")
    )
    # "blender" -> ``blender --background --python script``; "bpy" -> ``python script``.
    blender_mode: str = field(default_factory=lambda: _env("BLENDER_MODE", "blender") or "blender")
    blender_timeout_s: float = field(default_factory=lambda: float(_env("BLENDER_TIMEOUT", "600")))
    # Used when no GPU can be detected (e.g. a CPU-only dev box): total VRAM in GB.
    vram_override_gb: float | None = field(
        default_factory=lambda: float(v) if (v := _env("VRAM_GB")) else None
    )
    worker_concurrency: int = field(default_factory=lambda: int(_env("WORKERS", "2")))
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "3")))
    transient_retries: int = field(default_factory=lambda: int(_env("TRANSIENT_RETRIES", "1")))
    generation_timeout_s: float = field(
        default_factory=lambda: float(_env("GENERATION_TIMEOUT", "900"))
    )
    # Commercial use is assumed unless disabled; models whose license forbids it are filtered.
    commercial_use: bool = field(default_factory=lambda: _env_bool("COMMERCIAL_USE", True))
    thumbnails: bool = field(default_factory=lambda: _env_bool("THUMBNAILS", True))
    dedup_ttl_s: float = field(default_factory=lambda: float(_env("DEDUP_TTL", "3600")))
    # sqlite:///path, postgresql://..., or memory://; default is SQLite in the data dir.
    database_url: str | None = field(default_factory=lambda: _env("DATABASE_URL"))
    # Finished jobs, their files and unused uploads are deleted after this many days (0 = keep).
    retention_days: float = field(default_factory=lambda: float(_env("RETENTION_DAYS", "30")))

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def stats_path(self) -> Path:
        return self.data_dir / "model_stats.json"

    @property
    def run_log_path(self) -> Path:
        return self.data_dir / "runs.jsonl"

    @property
    def policies_path(self) -> Path:
        return self.data_dir / "policies.json"

    @property
    def blender_available(self) -> bool:
        return bool(self.blender_executable) and Path(self.blender_executable).exists()

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.jobs_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)
