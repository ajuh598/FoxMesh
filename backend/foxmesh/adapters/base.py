"""Adapter interface: how a registered model is actually executed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import ModelSpec, OutputFormat, TaskSpec


class TransientError(Exception):
    """A failure worth retrying on the same model (timeouts, connection resets)."""


@dataclass
class GenerationResult:
    path: Path
    format: OutputFormat
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationAdapter(ABC):
    #: Matches ``ModelSpec.adapter``.
    key: str

    def unavailable_reason(self, model: ModelSpec) -> str | None:
        """Why ``model`` cannot run with this adapter right now, or ``None`` if it can."""
        return None

    @abstractmethod
    async def generate(self, task: TaskSpec, model: ModelSpec, workdir: Path) -> GenerationResult:
        """Produce a 3D file for ``task`` inside ``workdir``."""
