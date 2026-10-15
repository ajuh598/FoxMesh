"""Task routing: pick the best model for a task with hard filters and weighted scores.

    score = w_acc * accuracy + w_speed * speed + w_quality * quality
          + w_stability * stability + w_resource * resource

All component scores live in [0, 1]. Hard constraints (task support, inputs,
output format, VRAM, licence, deadline, failure rate) are applied first: a model
that fails one is never selected, however good its score would be.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from .gpu import GpuMonitor
from .registry import ModelRegistry, blend
from .schemas import (
    BLENDER_IMPORTABLE,
    GENERATIVE_TASKS,
    Candidate,
    ModelSpec,
    ModelStats,
    Modality,
    Profile,
    Rejection,
    RoutingDecision,
    RoutingPolicy,
    Runtime,
    ScoreBreakdown,
    ScoreWeights,
    TaskSpec,
    Topology,
)

log = logging.getLogger(__name__)

DEFAULT_POLICIES: dict[Profile, RoutingPolicy] = {
    Profile.PROTOTYPE: RoutingPolicy(
        weights=ScoreWeights(accuracy=0.15, speed=0.35, quality=0.10, stability=0.15, resource=0.25),
        min_quality=0.0, reference_runtime_s=20,
    ),
    Profile.BALANCED: RoutingPolicy(
        weights=ScoreWeights(accuracy=0.25, speed=0.2, quality=0.25, stability=0.15, resource=0.15),
        min_quality=0.3, reference_runtime_s=60,
    ),
    Profile.HIGH_QUALITY: RoutingPolicy(
        weights=ScoreWeights(accuracy=0.3, speed=0.08, quality=0.37, stability=0.15, resource=0.1),
        min_quality=0.6, reference_runtime_s=300,
    ),
    Profile.GAME_ASSET: RoutingPolicy(
        weights=ScoreWeights(accuracy=0.25, speed=0.15, quality=0.35, stability=0.15, resource=0.1),
        min_quality=0.4, reference_runtime_s=90,
    ),
    Profile.PRINT_3D: RoutingPolicy(
        weights=ScoreWeights(accuracy=0.2, speed=0.1, quality=0.35, stability=0.3, resource=0.05),
        min_quality=0.4, reference_runtime_s=120,
    ),
}

# What "quality" means per profile: blend of geometry, texture, topology and watertightness.
QUALITY_MIX: dict[Profile, dict[str, float]] = {
    Profile.PROTOTYPE: {"mesh": 0.6, "texture": 0.4},
    Profile.BALANCED: {"mesh": 0.5, "texture": 0.4, "topology": 0.1},
    Profile.HIGH_QUALITY: {"mesh": 0.5, "texture": 0.5},
    Profile.GAME_ASSET: {"topology": 0.45, "mesh": 0.3, "texture": 0.25},
    Profile.PRINT_3D: {"watertight": 0.5, "mesh": 0.5},
}

TOPOLOGY_SCORE = {Topology.CLEAN: 1.0, Topology.DENSE: 0.5, Topology.SOUP: 0.2}

# A model is dropped once its recent failure rate exceeds this, given enough samples.
MAX_FAILURE_RATE = 0.5
MIN_SAMPLES_FOR_EXCLUSION = 5
# Measured runtimes count this much more than the prior when blending.
RUNTIME_SAFETY = 1.2


class PolicyStore:
    """Profile policies, defaulting to :data:`DEFAULT_POLICIES`, overridable and persisted."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._overrides: dict[Profile, RoutingPolicy] = {}
        if path and path.exists():
            try:
                raw = json.loads(path.read_text())
                self._overrides = {Profile(k): RoutingPolicy.model_validate(v) for k, v in raw.items()}
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable policy file %s: %s", path, exc)

    def get(self, profile: Profile) -> RoutingPolicy:
        return self._overrides.get(profile) or DEFAULT_POLICIES[profile]

    def all(self) -> dict[Profile, RoutingPolicy]:
        return {p: self.get(p) for p in Profile}

    def set(self, profile: Profile, policy: RoutingPolicy) -> None:
        self._overrides[profile] = policy
        self._persist()

    def reset(self, profile: Profile) -> None:
        self._overrides.pop(profile, None)
        self._persist()

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({p.value: v.model_dump() for p, v in self._overrides.items()}, indent=1)
        )


class NoCandidateError(Exception):
    def __init__(self, decision: RoutingDecision):
        self.decision = decision
        reasons = "; ".join(f"{r.model_id}: {', '.join(r.reasons)}" for r in decision.rejected)
        super().__init__(f"no model can run this task ({reasons or 'registry is empty'})")


def effective_runtime(spec: ModelSpec, stats: ModelStats) -> float:
    successes = stats.runs - stats.failures
    measured = stats.mean_runtime_s * RUNTIME_SAFETY if stats.mean_runtime_s else None
    return blend(spec.est_runtime_s, measured, successes)


def effective_accuracy(spec: ModelSpec, stats: ModelStats) -> float:
    return blend(spec.quality.accuracy, stats.mean_validation_score, stats.validated_runs)


def stability_score(stats: ModelStats) -> float:
    # Beta(1, 9) prior: an untested model is assumed to fail ~10% of the time.
    failures = stats.recent.count(False)
    return 1.0 - (failures + 1) / (len(stats.recent) + 10)


def quality_score(spec: ModelSpec, profile: Profile) -> float:
    parts = {
        "mesh": spec.quality.mesh,
        "texture": spec.quality.texture,
        "topology": TOPOLOGY_SCORE[spec.topology],
        "watertight": 1.0 if spec.watertight else 0.3,
    }
    mix = QUALITY_MIX[profile]
    return sum(parts[k] * w for k, w in mix.items()) / sum(mix.values())


def speed_score(runtime_s: float, reference_s: float) -> float:
    # Equals 0.5 at the reference runtime; logistic in log-time so 10x faster/slower
    # moves the score substantially but never saturates for realistic runtimes.
    return 1.0 / (1.0 + math.exp(1.2 * math.log(runtime_s / reference_s)))


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        gpu: GpuMonitor,
        policies: PolicyStore | None = None,
        blender_available: bool = False,
        max_fallbacks: int = 2,
    ):
        self.registry = registry
        self.gpu = gpu
        self.policies = policies or PolicyStore()
        self.blender_available = blender_available
        self.max_fallbacks = max_fallbacks

    # -- filtering -----------------------------------------------------------

    def _hard_constraints(
        self, spec: ModelSpec, task: TaskSpec, free_vram: float, total_vram: float
    ) -> tuple[list[str], bool]:
        """Return (rejection reasons, needs_conversion)."""
        reasons: list[str] = []
        if (why := self.registry.unavailable_reason(spec)) is not None:
            reasons.append(f"unavailable: {why}")
        if task.task_type not in spec.tasks:
            reasons.append(f"does not support {task.task_type.value}")

        generative_inputs = [m for m in task.inputs if m is not Modality.MESH]
        # Image-to-3D needs an image; a text prompt alongside it is optional context.
        required = [Modality.IMAGE] if Modality.IMAGE in generative_inputs else generative_inputs
        missing = [m.value for m in required if m not in spec.inputs]
        if missing:
            reasons.append(f"cannot take {', '.join(missing)} input")

        native = set(spec.outputs)
        wanted = set(task.output_formats)
        needs_conversion = not wanted <= native
        if needs_conversion:
            if not self.blender_available:
                reasons.append(
                    "cannot output " + ", ".join(sorted(f.value for f in wanted - native))
                    + " and Blender is not available for conversion"
                )
            elif not native & BLENDER_IMPORTABLE:
                reasons.append("output format cannot be imported into Blender")

        if spec.runtime is Runtime.LOCAL_GPU:
            if spec.vram_gb > total_vram:
                reasons.append(f"needs {spec.vram_gb:g} GB VRAM, GPU has {total_vram:g} GB")
            elif spec.vram_gb > free_vram:
                reasons.append(f"needs {spec.vram_gb:g} GB VRAM, {free_vram:g} GB free")

        if task.commercial_use and not spec.commercial_use:
            reasons.append(f"license does not allow commercial use ({spec.license})")

        stats = self.registry.stats.get(spec.id)
        rate = stats.recent_failure_rate
        if rate is not None and len(stats.recent) >= MIN_SAMPLES_FOR_EXCLUSION and rate > MAX_FAILURE_RATE:
            reasons.append(f"recent failure rate {rate:.0%}")

        if task.deadline_s is not None:
            runtime = effective_runtime(spec, stats)
            if runtime > task.deadline_s:
                reasons.append(f"expected {runtime:.0f}s exceeds deadline {task.deadline_s:.0f}s")
        return reasons, needs_conversion

    # -- scoring -------------------------------------------------------------

    def score(self, spec: ModelSpec, task: TaskSpec, policy: RoutingPolicy, free_vram: float) -> ScoreBreakdown:
        stats = self.registry.stats.get(spec.id)
        weights = policy.weights.normalized()
        runtime = effective_runtime(spec, stats)
        quality = quality_score(spec, task.profile)
        if task.require_textures and not spec.produces_textures:
            quality *= 0.8
        if spec.runtime is Runtime.LOCAL_GPU and spec.vram_gb > 0:
            resource = 1.0 - min(spec.vram_gb / max(free_vram, 1e-6), 1.0) * 0.9
        else:
            resource = 1.0
        parts = {
            "accuracy": effective_accuracy(spec, stats),
            "speed": speed_score(runtime, policy.reference_runtime_s),
            "quality": quality,
            "stability": stability_score(stats),
            "resource": resource,
        }
        total = sum(getattr(weights, k) * v for k, v in parts.items())
        return ScoreBreakdown(**{k: round(v, 4) for k, v in parts.items()}, total=round(total, 4))

    # -- routing -------------------------------------------------------------

    def route(self, task: TaskSpec) -> RoutingDecision:
        started = time.perf_counter()
        policy = self.policies.get(task.profile)
        gpu = self.gpu.snapshot()
        decision = RoutingDecision(
            profile=task.profile, task_type=task.task_type, selected=None,
            weights=policy.weights.normalized(), gpu=gpu,
            requires_model=task.task_type in GENERATIVE_TASKS,
        )
        if not decision.requires_model:
            # Conversion, preview and mesh edits run on Blender alone.
            if not self.blender_available:
                decision.warnings.append("Blender is not available; only basic file handling will run")
            decision.decided_in_ms = round((time.perf_counter() - started) * 1000, 3)
            return decision

        specs = self.registry.specs()
        if task.model_id:
            specs = [s for s in specs if s.id == task.model_id]
            if not specs:
                decision.rejected.append(Rejection(model_id=task.model_id, reasons=["unknown model"]))

        eligible: list[tuple[ModelSpec, bool]] = []
        for spec in specs:
            reasons, needs_conversion = self._hard_constraints(
                spec, task, gpu.free_vram_gb, gpu.total_vram_gb
            )
            if reasons:
                decision.rejected.append(Rejection(model_id=spec.id, reasons=reasons))
            else:
                eligible.append((spec, needs_conversion))

        candidates = []
        for spec, needs_conversion in eligible:
            scores = self.score(spec, task, policy, gpu.free_vram_gb)
            candidates.append(Candidate(
                model_id=spec.id, model_name=spec.name, scores=scores,
                est_runtime_s=round(effective_runtime(spec, self.registry.stats.get(spec.id)), 2),
                vram_gb=spec.vram_gb if spec.runtime is Runtime.LOCAL_GPU else 0.0,
                needs_conversion=needs_conversion,
            ))

        # Quality floor: keep only candidates meeting the profile's minimum quality,
        # unless that would leave nothing (then run the best we have and say so).
        meeting = [c for c in candidates if c.scores.quality >= policy.min_quality]
        if candidates and not meeting:
            decision.warnings.append(
                f"no model meets the {task.profile.value} quality floor "
                f"({policy.min_quality:.2f}); using the best available"
            )
            meeting = candidates
        elif len(meeting) < len(candidates):
            for c in candidates:
                if c not in meeting:
                    decision.rejected.append(Rejection(
                        model_id=c.model_id,
                        reasons=[f"quality {c.scores.quality:.2f} below floor {policy.min_quality:.2f}"],
                    ))

        ranked = sorted(meeting, key=lambda c: -c.scores.total)
        if ranked:
            # Among near-ties prefer the faster, then the lighter model.
            best = ranked[0].scores.total
            tied = [c for c in ranked if best - c.scores.total <= policy.tie_margin]
            tied.sort(key=lambda c: (c.est_runtime_s, c.vram_gb))
            ranked = tied + [c for c in ranked if c not in tied]
            decision.selected = ranked[0].model_id
            decision.fallbacks = [c.model_id for c in ranked[1 : 1 + self.max_fallbacks]]
        decision.candidates = ranked
        decision.decided_in_ms = round((time.perf_counter() - started) * 1000, 3)
        if decision.selected is None:
            raise NoCandidateError(decision)
        return decision
