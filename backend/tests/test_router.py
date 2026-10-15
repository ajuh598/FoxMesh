import pytest

from conftest import fixed_gpu, make_model
from foxmesh.router import NoCandidateError, PolicyStore, Router, speed_score
from foxmesh.schemas import (
    Modality, OutputFormat, Profile, QualityPriors, RoutingPolicy, ScoreWeights, TaskSpec, TaskType,
)


def task(profile=Profile.BALANCED, **kw) -> TaskSpec:
    data = dict(task_type=TaskType.GENERATE_3D, profile=profile, prompt="sword",
                inputs=[Modality.TEXT], output_formats=[OutputFormat.GLB])
    data.update(kw)
    return TaskSpec(**data)


def router(registry, gpu=None, blender=True):
    return Router(registry, gpu or fixed_gpu(24), PolicyStore(), blender_available=blender)


def reasons(decision, model_id):
    return next(r.reasons for r in decision.rejected if r.model_id == model_id)


def add_fast_and_good(registry):
    registry.register(make_model("fast", est_runtime_s=5, vram_gb=4,
                                 quality=QualityPriors(accuracy=0.45, mesh=0.45, texture=0.4)))
    registry.register(make_model("good", est_runtime_s=180, vram_gb=20,
                                 quality=QualityPriors(accuracy=0.85, mesh=0.9, texture=0.85)))


def test_profile_weights_change_the_choice(registry):
    add_fast_and_good(registry)
    r = router(registry)
    assert r.route(task(Profile.PROTOTYPE)).selected == "fast"
    hq = r.route(task(Profile.HIGH_QUALITY))
    assert hq.selected == "good"
    # "fast" misses the high-quality floor and is reported as such.
    assert any("below floor" in x for x in reasons(hq, "fast"))


def test_hard_filters_reject_with_reasons(registry):
    registry.register(make_model("ok"))
    registry.register(make_model("image-only", tasks=["image_to_3d"], inputs=["image"]))
    registry.register(make_model("huge", vram_gb=80))
    registry.register(make_model("nc", commercial_use=False, license="NC"))
    registry.register(make_model("off", enabled=False))
    registry.register(make_model("ply-only", outputs=["ply"]))
    d = router(registry, blender=False).route(task())
    assert d.selected == "ok"
    assert "does not support generate_3d" in reasons(d, "image-only")
    assert any("VRAM" in x for x in reasons(d, "huge"))
    assert any("commercial" in x for x in reasons(d, "nc"))
    assert reasons(d, "off") == ["unavailable: disabled"]
    assert any("Blender is not available" in x for x in reasons(d, "ply-only"))


def test_conversion_allowed_when_blender_available(registry):
    registry.register(make_model("ply-only", outputs=["ply"]))
    d = router(registry, blender=True).route(task(output_formats=[OutputFormat.GLB, OutputFormat.FBX]))
    assert d.selected == "ply-only"
    assert d.candidates[0].needs_conversion


def test_vram_uses_current_free_memory(registry):
    registry.register(make_model("big", vram_gb=16))
    registry.register(make_model("small", vram_gb=4, quality=QualityPriors(accuracy=0.3, mesh=0.4)))
    d = router(registry, gpu=fixed_gpu(24, free=10)).route(task())
    assert d.selected == "small"
    assert any("10 GB free" in x for x in reasons(d, "big"))


def test_vram_held_by_own_jobs_means_wait_not_reject(registry):
    import asyncio

    registry.register(make_model("big", vram_gb=16))
    gpu = fixed_gpu(24)

    async def scenario():
        async with gpu.reserve(12):
            # 12 GB free now, but 24 GB once our own job finishes: keep "big" and wait.
            d = router(registry, gpu=gpu).route(task())
            assert d.selected == "big"
            assert any("waits for 16 GB" in w for w in d.warnings)

    asyncio.run(scenario())


def test_remote_and_cpu_models_skip_vram_check(registry):
    registry.register(make_model("remote", runtime="remote", vram_gb=40))
    d = router(registry, gpu=fixed_gpu(0)).route(task())
    assert d.selected == "remote"


def test_failure_rate_excludes_model(registry):
    registry.register(make_model("flaky"))
    registry.register(make_model("steady", quality=QualityPriors(accuracy=0.4, mesh=0.4, texture=0.4)))
    for _ in range(6):
        registry.stats.record("flaky", ok=False, runtime_s=None, error="boom")
    d = router(registry).route(task())
    assert d.selected == "steady"
    assert any("in a row" in x and "cooling down" in x for x in reasons(d, "flaky"))


def test_breaker_reopens_for_a_trial_after_cooldown(registry):
    import time as _time

    from foxmesh.router import COOLDOWN_S

    registry.register(make_model("flaky"))
    for ok in (True, True, False, False, False):
        registry.stats.record("flaky", ok=ok, runtime_s=10 if ok else None)
    last = registry.stats.get("flaky").last_run_at
    r = router(registry)
    r.clock = lambda: last + 60
    with pytest.raises(NoCandidateError):
        r.route(task())
    # After the cooldown the model gets one trial run, flagged in the warnings.
    r.clock = lambda: last + COOLDOWN_S + 1
    d = r.route(task())
    assert d.selected == "flaky" and any("trial run" in w for w in d.warnings)
    # A success closes the breaker again.
    registry.stats.record("flaky", ok=True, runtime_s=10)
    r.clock = _time.time
    assert not router(registry).route(task()).warnings


def test_failure_rate_trips_without_a_streak(registry):
    registry.register(make_model("flaky"))
    registry.register(make_model("steady", quality=QualityPriors(accuracy=0.4, mesh=0.4, texture=0.4)))
    for ok in (False, True, False, False, True, False, False, True, False):
        registry.stats.record("flaky", ok=ok, runtime_s=1 if ok else None)
    d = router(registry).route(task())
    assert d.selected == "steady"
    assert any("failure rate" in x for x in reasons(d, "flaky"))


def test_measured_runtime_overrides_prior(registry):
    registry.register(make_model("a", est_runtime_s=10))
    registry.register(make_model("b", est_runtime_s=12))
    r = router(registry)
    assert r.route(task(Profile.PROTOTYPE)).selected == "a"
    for _ in range(10):
        registry.stats.record("a", ok=True, runtime_s=120, validation_score=1.0)
    assert r.route(task(Profile.PROTOTYPE)).selected == "b"


def test_deadline_filter(registry):
    registry.register(make_model("slow", est_runtime_s=300))
    with pytest.raises(NoCandidateError) as err:
        router(registry).route(task(deadline_s=60))
    assert "exceeds deadline" in str(err.value)


def test_quality_floor_relaxed_when_nothing_qualifies(registry):
    registry.register(make_model("meh", quality=QualityPriors(accuracy=0.3, mesh=0.3, texture=0.1)))
    d = router(registry).route(task(Profile.HIGH_QUALITY))
    assert d.selected == "meh"
    assert any("quality floor" in w for w in d.warnings)


def test_tie_prefers_faster_then_lighter(registry):
    registry.register(make_model("slow", est_runtime_s=31, vram_gb=8))
    registry.register(make_model("quick", est_runtime_s=30, vram_gb=8))
    policy = RoutingPolicy(weights=ScoreWeights(accuracy=1, speed=0, quality=0, stability=0, resource=0))
    r = router(registry)
    r.policies.set(Profile.BALANCED, policy)
    assert r.route(task()).selected == "quick"


def test_manual_model_choice_respects_constraints(registry):
    add_fast_and_good(registry)
    r = router(registry)
    assert r.route(task(Profile.PROTOTYPE, model_id="good")).selected == "good"
    with pytest.raises(NoCandidateError):
        r.route(task(model_id="missing"))


def test_fallbacks_are_ranked(registry):
    for i, q in enumerate([0.5, 0.7, 0.9]):
        registry.register(make_model(f"m{i}", quality=QualityPriors(accuracy=q, mesh=q, texture=q)))
    d = router(registry).route(task(Profile.HIGH_QUALITY))
    assert d.selected == "m2" and d.fallbacks == ["m1"]  # m0 is below the floor


def test_blender_only_tasks_need_no_model(registry):
    d = router(registry).route(task(task_type=TaskType.CONVERT, inputs=[Modality.MESH]))
    assert not d.requires_model and d.selected is None


def test_image_task_accepts_text_alongside(registry):
    registry.register(make_model("img", tasks=["image_to_3d"], inputs=["image"]))
    d = router(registry).route(task(task_type=TaskType.IMAGE_TO_3D,
                                    inputs=[Modality.TEXT, Modality.IMAGE]))
    assert d.selected == "img"


def test_scores_are_normalised(registry):
    add_fast_and_good(registry)
    for c in router(registry).route(task()).candidates:
        for value in c.scores.model_dump().values():
            assert 0.0 <= value <= 1.0


def test_speed_score_shape():
    assert speed_score(60, 60) == pytest.approx(0.5)
    assert speed_score(6, 60) > 0.9 > 0.1 > speed_score(600, 60)
