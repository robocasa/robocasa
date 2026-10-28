"""Exercise the formal-run smoke gate without allocating model or simulator GPUs."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from robocasa_eval import configuration, evaluation, gpu_worker, smoke


@pytest.fixture
def config(tmp_path):
    return configuration.build_config({"workspace": str(tmp_path / "workspace"),
        "envs_per_gpu": 4,
        "policy": {"factory": "robocasa_eval.examples.custom_policy:create_policy"},
    }, environment={})[0]


@pytest.fixture
def fake_execution(monkeypatch):
    calls = []
    def preflight(config, topo):
        config["runtime_identity"] = {"simulator": "fixture-version"}
        configuration.refresh_identity(config)
        return {"simulator": {"runtime_identity": config["runtime_identity"]}}
    def worker(config, rank, device):
        calls.append(copy.deepcopy(config))
        assert rank == 0
        assert config["total_gpus"] == config["gpus_per_node"] == 1
        assert Path.cwd() == Path(config["output_dir"])
        for job in evaluation.jobs_for(config):
            evaluation.atomic_json(evaluation.result_path(config["output_dir"], job), {
                "job_id": job["id"], "identity": config["identity"], "status": "timeout",
                "sim_seconds": job["timeout_seconds"], "wall_seconds": .01})
    monkeypatch.setattr(smoke, "preflight", preflight)
    monkeypatch.setattr(gpu_worker, "run", worker)
    return calls, preflight


def test_short_smoke_uses_actual_runner_but_cannot_authorize_full_run(config, fake_execution):
    calls, preflight = fake_execution
    original = copy.deepcopy(config)
    cwd = Path.cwd()
    report = smoke.run_smoke(config, envs=3, steps=25)
    assert report["status"] == "passed" and not report["full_horizons"]
    assert calls[0]["atomic_seconds"] == calls[0]["composite_seconds"] == .5
    assert report["groups"]["all"]["planned"] >= 3
    assert config == original and Path.cwd() == cwd
    preflight(config, {})
    with pytest.raises(ValueError, match="full-episodes"):
        evaluation.require_smoke(config)


def test_full_smoke_preserves_limits_and_authorizes_sixteen_gpus(config, fake_execution):
    calls, preflight = fake_execution
    report = smoke.run_smoke(config, envs=4, full_episodes=True)
    tested = calls[0]
    assert (tested["atomic_seconds"], tested["composite_seconds"]) == (40, 180)
    assert report["scorable_complete"] and report["groups"]["all"]["timeout"] == 4
    assert report["engineering_test"]
    preflight(config, {})
    receipt = evaluation.require_smoke(config)
    assert receipt["tested_environment_count"] == 4
    assert receipt["tested_categories"] == ["atomic", "composite"]
    assert Path(receipt["report"]).is_file()
    assert config["total_gpus"] == 16


@pytest.mark.parametrize("field,value", [("camera_names", ["ego_view", "right_wrist"]),
    ("decoder_sha256", "different-decoder"), ("execution_horizon", 10),
    ("composite_seconds", 150), ("runtime_identity", {"simulator": "changed"}),
    ("policy_identity", {"model": "different-model"})])
def test_protocol_changes_require_new_smoke(config, fake_execution, field, value):
    _, preflight = fake_execution
    smoke.run_smoke(config, envs=2, full_episodes=True)
    preflight(config, {})
    evaluation.require_smoke(config)
    config[field] = value
    with pytest.raises(ValueError, match="full-episodes"):
        evaluation.require_smoke(config)


@pytest.mark.parametrize("kwargs", [dict(envs=1, full_episodes=True),
    dict(envs=2, full_episodes=True, tasks=["CloseDishwasher"]),
    dict(tasks=["MissingTask"]), dict(tasks=["OpenDrawer", "OpenDrawer"])])
def test_inadequate_smoke_rejected_before_execution(config, fake_execution, kwargs):
    calls, _ = fake_execution
    with pytest.raises(ValueError):
        smoke.run_smoke(config, **kwargs)
    assert not calls


@pytest.mark.parametrize("failure", ["error", "pending", "interrupt"])
def test_failed_smoke_never_writes_receipt_and_restores_cwd(config, fake_execution, monkeypatch, failure):
    _, preflight = fake_execution
    cwd = Path.cwd()
    def failed_worker(smoke_config, *args):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure == "error":
            for job in evaluation.jobs_for(smoke_config):
                evaluation.atomic_json(evaluation.result_path(smoke_config["output_dir"], job), {
                    "job_id": job["id"], "identity": smoke_config["identity"], "status": "error"})
    monkeypatch.setattr(gpu_worker, "run", failed_worker)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        smoke.run_smoke(config, envs=4, full_episodes=True)
    assert Path.cwd() == cwd
    assert list((Path(config["workspace"]) / "eval_smoke").glob("*/failure.json"))
    preflight(config, {})
    with pytest.raises(ValueError):
        evaluation.require_smoke(config)


def test_simulator_probe_runtime_identity_is_bound_before_gate(config, tmp_path, monkeypatch):
    decoder = tmp_path / "decoder.onnx"
    decoder.write_bytes(b"fixture")
    import sys
    config.update(sim_python=sys.executable, decoder_path=str(decoder))
    monkeypatch.setattr(evaluation, "prepare_workspace", lambda c: None)
    monkeypatch.setattr(evaluation, "disk_preflight", lambda c: {})
    runtime = {"sources": {"robocasa": "version-one"}, "versions": {"mujoco": "fixture"}}
    def subprocess_run(command, **kwargs):
        assert command[2:4] == ["-m", "robocasa_eval.sim_worker"]
        assert Path(kwargs["cwd"]).is_relative_to(Path(config["workspace"]))
        return subprocess.CompletedProcess(command, 0, "Simulator startup banner\n" + json.dumps({
            "runtime_identity": runtime, "render_probe": "passed"}) + "\n", "")
    monkeypatch.setattr(evaluation.subprocess, "run", subprocess_run)
    original = config["identity"]
    evaluation.preflight(config, {"local": 1})
    assert config["runtime_identity"] == runtime and config["identity"] != original
    first = config["identity"]
    runtime["sources"]["robocasa"] = "version-two"
    evaluation.preflight(config, {"local": 1})
    assert config["identity"] != first
