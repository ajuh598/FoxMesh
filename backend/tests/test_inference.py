import httpx

from foxmesh.adapters import HttpInferenceAdapter, TransientError
from foxmesh.inference.server import ModelRunner, ProceduralRunner, RunnerPool, create_inference_app
from foxmesh.schemas import Modality, ModelSpec, OutputFormat, Profile, TaskSpec, TaskType


def spec(**config):
    return ModelSpec(id="proc", name="proc", version="1", adapter="http", tasks=["generate_3d"],
                     inputs=["text"], outputs=["obj"], runtime="remote",
                     config={"endpoint": "http://inference", "model": "procedural", **config})


def task():
    return TaskSpec(task_type=TaskType.GENERATE_3D, profile=Profile.BALANCED, prompt="sword",
                    inputs=[Modality.TEXT], output_formats=[OutputFormat.OBJ])


async def test_http_adapter_against_inference_server(tmp_path):
    app = create_inference_app(RunnerPool([ProceduralRunner()]))
    adapter = HttpInferenceAdapter(transport=httpx.ASGITransport(app=app))
    result = await adapter.generate(task(), spec(), tmp_path)
    assert result.format is OutputFormat.OBJ
    assert result.path.read_text().startswith("# FoxMesh")


async def test_unknown_model_is_an_error(tmp_path):
    app = create_inference_app(RunnerPool([ProceduralRunner()]))
    adapter = HttpInferenceAdapter(transport=httpx.ASGITransport(app=app))
    s = spec()
    s.config["model"] = "missing"
    try:
        await adapter.generate(task(), s, tmp_path)
    except RuntimeError as exc:
        assert "404" in str(exc)
    else:
        raise AssertionError("expected failure")


async def test_busy_server_is_transient(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    adapter = HttpInferenceAdapter(transport=transport)
    try:
        await adapter.generate(task(), spec(), tmp_path)
    except TransientError:
        pass
    else:
        raise AssertionError("expected TransientError")


def test_adapter_requires_endpoint():
    assert HttpInferenceAdapter().unavailable_reason(spec(endpoint=None))


class Dummy(ModelRunner):
    output_formats = ("obj",)

    def __init__(self, name, vram):
        super().__init__()
        self.name, self.vram_gb, self.unloads = name, vram, 0

    def load(self):
        pass

    def unload(self):
        self.unloads += 1

    def generate(self, request, out_path):
        return out_path


async def test_pool_keeps_models_loaded_and_evicts_lru():
    a, b, c = Dummy("a", 10), Dummy("b", 10), Dummy("c", 10)
    pool = RunnerPool([a, b, c], vram_budget_gb=20)
    await pool.acquire("a")
    await pool.acquire("b")
    await pool.acquire("a")  # reuse: no reload, no eviction
    assert a.loaded and b.loaded and a.unloads == b.unloads == 0
    a.last_used, b.last_used = 2.0, 1.0
    await pool.acquire("c")
    assert not b.loaded and b.unloads == 1 and a.loaded and c.loaded
