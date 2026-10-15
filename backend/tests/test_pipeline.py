import asyncio
from pathlib import Path

from conftest import fixed_gpu, make_model
from foxmesh.adapters import AdapterRegistry, GenerationAdapter, GenerationResult, TransientError
from foxmesh.adapters.procedural import generate_mesh
from foxmesh.blender import LitePostProcessor
from foxmesh.pipeline import Job, JobStatus, Pipeline
from foxmesh.router import PolicyStore, Router
from foxmesh.schemas import JobRequest, Modality, OutputFormat, Profile, QualityPriors, TaskSpec, TaskType


class FakeAdapter(GenerationAdapter):
    """Behaviour per model comes from ``config.behavior``: a list consumed one call at a time."""

    key = "fake"

    def __init__(self):
        self.calls: list[str] = []

    async def generate(self, task, model, workdir: Path) -> GenerationResult:
        self.calls.append(model.id)
        script = model.config.setdefault("behavior", ["ok"])
        action = script.pop(0) if len(script) > 1 else script[0]
        if action == "transient":
            raise TransientError("server busy")
        if action == "fail":
            raise RuntimeError("CUDA error")
        if action == "slow":
            await asyncio.sleep(10)
        path = workdir / "out.obj"
        if action == "garbage":
            path.write_bytes(b"\x00\x01garbage")
        else:
            generate_mesh("cube")[1].write_obj(path)
        return GenerationResult(path=path, format=OutputFormat.OBJ)


def setup(registry, tmp_path, models, **pipeline_kw):
    adapter = FakeAdapter()
    for spec in models:
        registry.register(spec)
    gpu = fixed_gpu(24)
    router = Router(registry, gpu, PolicyStore(), blender_available=False)
    pipeline = Pipeline(registry, AdapterRegistry([adapter], load_plugins=False), router, gpu,
                        LitePostProcessor(), tmp_path, **pipeline_kw)
    return pipeline, adapter


def job() -> Job:
    spec = TaskSpec(task_type=TaskType.GENERATE_3D, profile=Profile.BALANCED, prompt="cube",
                    inputs=[Modality.TEXT], output_formats=[OutputFormat.OBJ])
    return Job(id="j1", request=JobRequest(prompt="cube"), spec=spec, fingerprint=spec.fingerprint())


def model(model_id, quality, behavior, **kw):
    return make_model(model_id, outputs=["obj"], config={"behavior": behavior},
                      quality=QualityPriors(accuracy=quality, mesh=quality, texture=quality), **kw)


async def test_success_records_stats(registry, tmp_path):
    pipeline, adapter = setup(registry, tmp_path, [model("a", 0.8, ["ok"])])
    result = await pipeline.run(job())
    assert result.status is JobStatus.SUCCEEDED, result.error
    assert result.result.model_id == "a" and result.result.files == {"obj": "model.obj"}
    assert (Path(result.result.result_dir) / "model.obj").exists()
    stats = registry.stats.get("a")
    assert stats.runs == 1 and stats.mean_validation_score == 1.0
    assert (tmp_path / "j1" / "job.json").exists()


async def test_falls_back_to_next_model(registry, tmp_path):
    pipeline, adapter = setup(registry, tmp_path, [model("best", 0.9, ["fail"]),
                                                   model("second", 0.7, ["ok"])])
    result = await pipeline.run(job())
    assert result.status is JobStatus.SUCCEEDED
    assert adapter.calls == ["best", "second"]
    assert result.result.model_id == "second"
    assert registry.stats.get("best").failures == 1


async def test_transient_error_retries_same_model_once(registry, tmp_path):
    pipeline, adapter = setup(registry, tmp_path, [model("a", 0.9, ["transient", "ok"]),
                                                   model("b", 0.5, ["ok"])])
    result = await pipeline.run(job())
    assert adapter.calls == ["a", "a"] and result.result.model_id == "a"


async def test_validation_failure_moves_on(registry, tmp_path):
    pipeline, adapter = setup(registry, tmp_path, [model("a", 0.9, ["garbage"]),
                                                   model("b", 0.6, ["ok"])])
    result = await pipeline.run(job())
    assert result.result.model_id == "b"
    assert "validation failed" in result.attempts[0].error


async def test_attempts_are_capped(registry, tmp_path):
    models = [model(f"m{i}", 0.9 - i * 0.1, ["transient"]) for i in range(3)]
    pipeline, adapter = setup(registry, tmp_path, models, max_attempts=3, transient_retries=1)
    result = await pipeline.run(job())
    assert result.status is JobStatus.FAILED
    assert len(adapter.calls) == 3 and "all 3 attempts failed" in result.error


async def test_no_candidate_fails_with_routing_info(registry, tmp_path):
    pipeline, _ = setup(registry, tmp_path, [model("huge", 0.9, ["ok"], vram_gb=99)])
    result = await pipeline.run(job())
    assert result.status is JobStatus.FAILED
    assert result.routing.rejected[0].model_id == "huge"


async def test_generation_timeout_is_transient(registry, tmp_path):
    pipeline, adapter = setup(registry, tmp_path, [model("a", 0.9, ["slow"])],
                              generation_timeout_s=0.05, transient_retries=0)
    result = await pipeline.run(job())
    assert result.status is JobStatus.FAILED and result.attempts[0].transient
