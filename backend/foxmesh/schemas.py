"""Structured data exchanged between the analyzer, router, workers and API.

Every stage of the pipeline talks in these types instead of natural language:
the request analyzer produces a :class:`TaskSpec`, the router turns it into a
:class:`RoutingDecision`, and workers report back with structured reports.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    GENERATE_3D = "generate_3d"  # text -> 3D
    IMAGE_TO_3D = "image_to_3d"  # image (+ optional text) -> 3D
    EDIT_3D = "edit_3d"  # optimise / repair an existing mesh
    TEXTURE = "texture"  # check / attach materials and textures on an existing mesh
    CONVERT = "convert"  # format conversion only
    PREVIEW = "preview"  # thumbnail render of an existing mesh


# Task types that need a generative AI model; the rest run on Blender alone.
GENERATIVE_TASKS = frozenset({TaskType.GENERATE_3D, TaskType.IMAGE_TO_3D})


class Profile(str, Enum):
    PROTOTYPE = "prototype"  # speed and resource efficiency first
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"  # quality and accuracy first
    GAME_ASSET = "game_asset"  # mesh structure and polygon efficiency first
    PRINT_3D = "print_3d"  # shape stability and mesh checks first


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MESH = "mesh"


class OutputFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"
    FBX = "fbx"
    STL = "stl"
    PLY = "ply"
    BLEND = "blend"


# Formats Blender can import, i.e. formats a model may emit and still be converted.
BLENDER_IMPORTABLE = frozenset(
    {OutputFormat.GLB, OutputFormat.OBJ, OutputFormat.FBX, OutputFormat.STL, OutputFormat.PLY}
)


class Topology(str, Enum):
    CLEAN = "clean"  # quad-ish, low/medium density, game friendly
    DENSE = "dense"  # marching-cubes style dense triangles
    SOUP = "soup"  # unstructured triangles, may be non-manifold


class Runtime(str, Enum):
    LOCAL_GPU = "local_gpu"  # consumes VRAM on this host
    REMOTE = "remote"  # inference server with its own GPU
    CPU = "cpu"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class JobRequest(BaseModel):
    """What the web client submits. Only ``prompt`` (or an input file) is required."""

    prompt: str = ""
    image_id: str | None = Field(None, description="Upload id of a reference image")
    input_mesh_id: str | None = Field(None, description="Upload id of an existing 3D file")
    profile: Profile | None = Field(None, description="Override the detected profile")
    output_formats: list[OutputFormat] | None = None
    max_polygons: int | None = Field(None, gt=0)
    target_size_m: float | None = Field(None, gt=0)
    deadline_s: float | None = Field(None, gt=0)
    model_id: str | None = Field(None, description="Manual model choice (advanced users)")


class TaskSpec(BaseModel):
    """Result of request analysis: everything the router and workers need."""

    task_type: TaskType
    profile: Profile
    prompt: str = ""
    inputs: list[Modality] = Field(default_factory=list)
    image_path: str | None = None
    input_mesh_path: str | None = None
    output_formats: list[OutputFormat] = Field(default_factory=lambda: [OutputFormat.GLB])
    max_polygons: int | None = None
    target_size_m: float | None = None
    require_textures: bool = False
    require_watertight: bool = False
    deadline_s: float | None = None
    model_id: str | None = None
    commercial_use: bool = True
    # Which rules fired during analysis, for display in the UI.
    matched_rules: list[str] = Field(default_factory=list)

    def fingerprint(self) -> str:
        """Stable hash used to detect duplicate requests."""
        payload = self.model_dump(mode="json", exclude={"matched_rules"})
        payload["prompt"] = " ".join(self.prompt.lower().split())
        for key in ("image_path", "input_mesh_path"):
            if payload.get(key):
                payload[key] = _file_digest(payload[key])
        payload["output_formats"] = sorted(payload["output_formats"])
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


def _file_digest(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return path


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class QualityPriors(BaseModel):
    """Prior quality estimates in [0, 1]; replaced by measurements as runs accumulate."""

    accuracy: float = Field(0.5, ge=0, le=1, description="How well results match the request")
    mesh: float = Field(0.5, ge=0, le=1, description="Geometry quality")
    texture: float = Field(0.0, ge=0, le=1, description="Texture/material quality")


class ModelSpec(BaseModel):
    id: str
    name: str
    version: str
    adapter: str = Field(description="Adapter key that knows how to run this model")
    tasks: list[TaskType]
    inputs: list[Modality]
    outputs: list[OutputFormat]
    runtime: Runtime = Runtime.LOCAL_GPU
    vram_gb: float = Field(0.0, ge=0)
    est_runtime_s: float = Field(60.0, gt=0)
    quality: QualityPriors = Field(default_factory=QualityPriors)
    topology: Topology = Topology.DENSE
    watertight: bool = False
    produces_textures: bool = False
    license: str = "unknown"
    commercial_use: bool = False
    enabled: bool = True
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class ModelStats(BaseModel):
    """Measured behaviour of a model, from real runs."""

    runs: int = 0
    failures: int = 0
    validated_runs: int = 0
    mean_runtime_s: float | None = None
    mean_validation_score: float | None = None
    # Most recent outcomes, newest last (True = success).
    recent: list[bool] = Field(default_factory=list)
    last_error: str | None = None
    last_run_at: float | None = None

    @property
    def recent_failure_rate(self) -> float | None:
        if not self.recent:
            return None
        return self.recent.count(False) / len(self.recent)


class ModelView(BaseModel):
    spec: ModelSpec
    stats: ModelStats
    available: bool
    unavailable_reason: str | None = None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class ScoreWeights(BaseModel):
    accuracy: float = 0.2
    speed: float = 0.2
    quality: float = 0.2
    stability: float = 0.2
    resource: float = 0.2

    def normalized(self) -> "ScoreWeights":
        total = sum(self.model_dump().values())
        if total <= 0:
            return ScoreWeights()
        return ScoreWeights(**{k: v / total for k, v in self.model_dump().items()})


class RoutingPolicy(BaseModel):
    weights: ScoreWeights
    # Candidates whose quality score is below this are dropped (unless none remain).
    min_quality: float = Field(0.0, ge=0, le=1)
    # Runtime at which the speed score is 0.5; shorter runs score higher.
    reference_runtime_s: float = Field(60.0, gt=0)
    # Scores within this margin of the best count as a tie, broken by speed then VRAM.
    tie_margin: float = Field(0.02, ge=0)


class ScoreBreakdown(BaseModel):
    accuracy: float
    speed: float
    quality: float
    stability: float
    resource: float
    total: float


class Candidate(BaseModel):
    model_id: str
    model_name: str
    scores: ScoreBreakdown
    est_runtime_s: float
    vram_gb: float
    needs_conversion: bool


class Rejection(BaseModel):
    model_id: str
    reasons: list[str]


class GpuSnapshot(BaseModel):
    detected: bool
    total_vram_gb: float
    free_vram_gb: float
    reserved_vram_gb: float = 0.0
    devices: list[dict[str, Any]] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    profile: Profile
    task_type: TaskType
    selected: str | None
    fallbacks: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    weights: ScoreWeights
    warnings: list[str] = Field(default_factory=list)
    gpu: GpuSnapshot
    decided_in_ms: float = 0.0
    requires_model: bool = True
