import json

from conftest import make_model
from foxmesh.registry import ModelRegistry, StatsStore, blend


def test_blend_moves_from_prior_to_measurement():
    assert blend(10, None, 0) == 10
    assert blend(10, 20, 3) == 15
    assert abs(blend(10, 20, 300) - 20) < 0.2


def test_stats_record_and_persist(tmp_path):
    path, log = tmp_path / "stats.json", tmp_path / "runs.jsonl"
    store = StatsStore(path, log)
    store.record("m", ok=True, runtime_s=10, validation_score=1.0)
    store.record("m", ok=True, runtime_s=20, validation_score=0.5)
    store.record("m", ok=False, runtime_s=None, error="boom")
    s = store.get("m")
    assert (s.runs, s.failures, s.validated_runs) == (3, 1, 2)
    assert s.mean_runtime_s == 15 and s.mean_validation_score == 0.75
    assert s.recent == [True, True, False] and s.last_error == "boom"
    assert len(log.read_text().splitlines()) == 3
    reloaded = StatsStore(path).get("m")
    assert reloaded.runs == 3 and reloaded.mean_runtime_s == 15


def test_builtin_descriptors_load():
    reg = ModelRegistry(StatsStore())
    assert reg.load_builtin() >= 5
    assert reg.get("procedural-primitives").enabled


def test_update_merges_config_and_bumps_revision(registry):
    registry.register(make_model("m", config={"endpoint": None, "steps": 20}))
    rev = registry.revision
    spec = registry.update("m", config={"endpoint": "http://gpu:9000"}, enabled=False)
    assert spec.config == {"endpoint": "http://gpu:9000", "steps": 20}
    assert not spec.enabled and registry.revision > rev


def test_plugin_directory(tmp_path, registry):
    (tmp_path / "custom.json").write_text(make_model("custom").model_dump_json())
    (tmp_path / "broken.json").write_text(json.dumps({"id": "x"}))
    assert registry.load_dir(tmp_path) == 1
    assert registry.get("custom") is not None


def test_availability_callback(registry):
    registry.register(make_model("m"))
    registry.set_availability(lambda spec: "no endpoint")
    view = registry.view("m")
    assert not view.available and view.unavailable_reason == "no endpoint"
