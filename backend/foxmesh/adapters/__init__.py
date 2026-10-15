"""Adapter registry. Third-party adapters register via the ``foxmesh.adapters`` entry point."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from ..schemas import ModelSpec
from .base import GenerationAdapter, GenerationResult, TransientError
from .http import HttpInferenceAdapter
from .image_relief import ImageReliefAdapter
from .procedural import ProceduralAdapter

log = logging.getLogger(__name__)

__all__ = [
    "AdapterRegistry", "GenerationAdapter", "GenerationResult", "TransientError",
    "HttpInferenceAdapter", "ImageReliefAdapter", "ProceduralAdapter",
]


class AdapterRegistry:
    def __init__(self, adapters: list[GenerationAdapter] | None = None, load_plugins: bool = True):
        self._adapters: dict[str, GenerationAdapter] = {}
        for adapter in adapters if adapters is not None else [ProceduralAdapter(), ImageReliefAdapter(), HttpInferenceAdapter()]:
            self.add(adapter)
        if load_plugins:
            self._load_entry_points()

    def add(self, adapter: GenerationAdapter) -> None:
        self._adapters[adapter.key] = adapter

    def get(self, key: str) -> GenerationAdapter | None:
        return self._adapters.get(key)

    def keys(self) -> list[str]:
        return sorted(self._adapters)

    def unavailable_reason(self, model: ModelSpec) -> str | None:
        """Availability check plugged into :class:`~foxmesh.registry.ModelRegistry`."""
        adapter = self._adapters.get(model.adapter)
        if adapter is None:
            return f"no adapter '{model.adapter}' installed"
        return adapter.unavailable_reason(model)

    def _load_entry_points(self) -> None:
        for ep in entry_points(group="foxmesh.adapters"):
            try:
                factory = ep.load()
                self.add(factory())
            except Exception as exc:  # a broken plugin must not take the service down
                log.error("Failed to load adapter plugin %s: %s", ep.name, exc)
