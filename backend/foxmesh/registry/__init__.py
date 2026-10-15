"""Central registry of AI models / tools and their measured performance."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..schemas import ModelSpec, ModelStats, ModelView

log = logging.getLogger(__name__)

BUILTIN_MODELS_DIR = Path(__file__).parent / "models"
RECENT_WINDOW = 20
# Prior strength: how many real runs a prior estimate is worth when blending.
PRIOR_WEIGHT = 3.0


def blend(prior: float, measured: float | None, samples: int, weight: float = PRIOR_WEIGHT) -> float:
    """Shrink a measured mean toward its prior; measurements dominate as samples grow."""
    if measured is None or samples <= 0:
        return prior
    return (prior * weight + measured * samples) / (weight + samples)


class StatsStore:
    """Per-model run statistics persisted as JSON, plus an append-only JSONL run log."""

    def __init__(self, path: Path | None = None, run_log: Path | None = None):
        self._path = path
        self._run_log = run_log
        self._lock = threading.Lock()
        self._stats: dict[str, ModelStats] = {}
        if path and path.exists():
            try:
                raw = json.loads(path.read_text())
                self._stats = {k: ModelStats.model_validate(v) for k, v in raw.items()}
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable stats file %s: %s", path, exc)

    def get(self, model_id: str) -> ModelStats:
        with self._lock:
            return self._stats.get(model_id, ModelStats()).model_copy(deep=True)

    def record(
        self,
        model_id: str,
        ok: bool,
        runtime_s: float | None,
        validation_score: float | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ModelStats:
        with self._lock:
            stats = self._stats.setdefault(model_id, ModelStats())
            stats.runs += 1
            if not ok:
                stats.failures += 1
                stats.last_error = error
            stats.recent = (stats.recent + [ok])[-RECENT_WINDOW:]
            if ok and runtime_s is not None:
                # Only successful runs say how long the model takes.
                n = stats.runs - stats.failures
                prev = stats.mean_runtime_s
                stats.mean_runtime_s = runtime_s if prev is None else prev + (runtime_s - prev) / n
            if validation_score is not None:
                stats.validated_runs += 1
                prev = stats.mean_validation_score
                stats.mean_validation_score = (
                    validation_score
                    if prev is None
                    else prev + (validation_score - prev) / stats.validated_runs
                )
            stats.last_run_at = time.time()
            snapshot = stats.model_copy(deep=True)
            self._persist_locked()
            self._log_locked(
                {
                    "ts": stats.last_run_at,
                    "model_id": model_id,
                    "ok": ok,
                    "runtime_s": runtime_s,
                    "validation_score": validation_score,
                    "error": error,
                    **(extra or {}),
                }
            )
        return snapshot

    def reset(self, model_id: str) -> None:
        with self._lock:
            self._stats.pop(model_id, None)
            self._persist_locked()

    def _persist_locked(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.model_dump() for k, v in self._stats.items()}, indent=1))
        os.replace(tmp, self._path)

    def _log_locked(self, entry: dict[str, Any]) -> None:
        if not self._run_log:
            return
        self._run_log.parent.mkdir(parents=True, exist_ok=True)
        with self._run_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


class ModelRegistry:
    """Holds model descriptors; models are plain JSON files, so adding one needs no code change.

    ``availability`` is a callback (usually the adapter registry) telling whether a
    model can actually run right now, e.g. whether its inference endpoint is configured.
    """

    def __init__(
        self,
        stats: StatsStore | None = None,
        availability: Callable[[ModelSpec], str | None] | None = None,
    ):
        self.stats = stats or StatsStore()
        self._availability = availability
        self._models: dict[str, ModelSpec] = {}
        self._lock = threading.Lock()
        self.revision = 0

    # -- loading -----------------------------------------------------------

    def load_dir(self, directory: Path) -> int:
        count = 0
        for path in sorted(directory.glob("*.json")):
            try:
                self.register(ModelSpec.model_validate_json(path.read_text()))
                count += 1
            except (OSError, ValueError) as exc:
                log.error("Skipping invalid model descriptor %s: %s", path, exc)
        return count

    def load_builtin(self) -> int:
        return self.load_dir(BUILTIN_MODELS_DIR)

    def set_availability(self, check: Callable[[ModelSpec], str | None]) -> None:
        self._availability = check

    # -- CRUD ----------------------------------------------------------------

    def register(self, spec: ModelSpec) -> None:
        with self._lock:
            self._models[spec.id] = spec
            self.revision += 1

    def unregister(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)
            self.revision += 1

    def get(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def update(self, model_id: str, **changes: Any) -> ModelSpec:
        with self._lock:
            current = self._models.get(model_id)
            if current is None:
                raise KeyError(model_id)
            data = current.model_dump()
            for key, value in changes.items():
                if value is None:
                    continue
                if key == "config":
                    data["config"] = {**data["config"], **value}
                elif key == "quality":
                    data["quality"] = {**data["quality"], **value}
                else:
                    data[key] = value
            updated = ModelSpec.model_validate(data)
            self._models[model_id] = updated
            self.revision += 1
            return updated

    def specs(self) -> list[ModelSpec]:
        return list(self._models.values())

    # -- views ---------------------------------------------------------------

    def unavailable_reason(self, spec: ModelSpec) -> str | None:
        if not spec.enabled:
            return "disabled"
        if self._availability:
            return self._availability(spec)
        return None

    def view(self, model_id: str) -> ModelView | None:
        spec = self.get(model_id)
        if spec is None:
            return None
        reason = self.unavailable_reason(spec)
        return ModelView(
            spec=spec, stats=self.stats.get(model_id), available=reason is None,
            unavailable_reason=reason,
        )

    def views(self) -> list[ModelView]:
        return [v for spec in self.specs() if (v := self.view(spec.id))]
