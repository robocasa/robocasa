"""Service configuration, remote provenance, and simulation-free CLI checks."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

from robocasa_eval import configuration, evaluation, serving_configuration as serving


@pytest.fixture
def service_options(tmp_path):
    return dict(workspace=str(tmp_path / "service"), devices=["cpu"],
                minimum_free_bytes=0, policy={
                    "factory": "robocasa_eval.examples.custom_policy:create_policy",
                    "id": "interface-fixture", "kwargs": {}, "artifacts": []})


def test_model_service_configuration_needs_no_simulator_assets(service_options):
    config = serving.build_server_config(service_options, verify_files=True)
    assert config["port"] == 8000
    assert "sim_python" not in config and "decoder_path" not in config
    assert config["policy_identity"]["verified"] is True
    metadata = serving.service_metadata(config)
    assert metadata["model_id"] == "interface-fixture"
    assert metadata["replicas"] == 1 and len(metadata["fingerprint"]) == 64
    assert metadata["action_dims"] == dict(motion_token=64, left_hand_joints=7, right_hand_joints=7)


@pytest.mark.parametrize("update", [{"port": 0}, {"port": 65536}, {"port": True},
    {"devices": []}, {"devices": ["cuda:0", "cuda:0"]}, {"devices": [[]]},
    {"devices": ["cuda"]}, {"request_timeout_seconds": float("nan")},
    {"idle_session_seconds": 0}, {"max_batch_size": False}, {"max_request_bytes": 128 * 1024**2},
    {"auth_token_env": "secret\nvalue"}, {"policy": {"factory": "http", "id": "v1"}},
    {"sim_python": sys.executable}, {"workspace": None}, {"host": ""}])
def test_invalid_service_configuration_rejected(service_options, update):
    with pytest.raises(ValueError):
        serving.build_server_config(dict(service_options, **update))


def test_service_requires_explicit_model():
    with pytest.raises(ValueError, match="explicit policy"):
        serving.build_server_config({})


def test_single_device_smoke_model_identity_survives_sixteen_device_deployment(service_options):
    first = serving.build_server_config(dict(service_options, devices=["cuda:0"]), verify_files=True)
    scaled = serving.build_server_config(dict(service_options,
        devices=[f"cuda:{n}" for n in range(16)], port=8080), verify_files=True)
    a, b = serving.service_metadata(first), serving.service_metadata(scaled)
    assert a["fingerprint"] == b["fingerprint"]
    assert (a["replicas"], b["replicas"]) == (1, 16)
    modified = copy.deepcopy(scaled)
    modified["policy"]["id"] = "a-different-model"
    assert serving.service_metadata(modified)["fingerprint"] != a["fingerprint"]


def client_config(tmp_path):
    return configuration.build_config(dict(workspace=str(tmp_path / "client"),
        num_workers=16, envs_per_worker=4, device_prefix="cpu",
        policy={"factory": "http", "id": "interface-fixture", "kwargs": {
            "endpoint": "http://127.0.0.1:8000"}}), environment={})[0]


def test_cpu_smoke_identity_cannot_approve_cuda_model_service(service_options):
    cpu = serving.build_server_config(service_options, verify_files=True)
    cuda = serving.build_server_config(dict(service_options, devices=["cuda:0"]), verify_files=True)
    assert serving.service_metadata(cpu)["fingerprint"] != serving.service_metadata(cuda)["fingerprint"]


def test_remote_metadata_pins_model_and_gate_independent_of_replica_count(service_options, tmp_path, monkeypatch):
    from robocasa_eval.adapters import http
    metadata = serving.service_metadata(serving.build_server_config(service_options))
    monkeypatch.setattr(http, "fetch_metadata", lambda *a, **kw: dict(metadata))
    first = client_config(tmp_path)
    serving.bind_remote_policy(first)
    assert first["policy"]["kwargs"]["model_id"] == first["policy"]["id"]
    assert first["policy"]["kwargs"]["fingerprint"] == metadata["fingerprint"]
    assert first["remote_policy_identity"]["model_id"] == metadata["model_id"]
    assert "replicas" not in first["remote_policy_identity"]
    metadata["replicas"] = 16
    scaled = client_config(tmp_path)
    serving.bind_remote_policy(scaled)
    assert configuration.gate_protocol(first) == configuration.gate_protocol(scaled)
    metadata["fingerprint"] = "changed-checkpoint"
    changed = client_config(tmp_path)
    serving.bind_remote_policy(changed)
    assert configuration.gate_protocol(first) != configuration.gate_protocol(changed)
    with pytest.raises(ValueError, match="fingerprint changed"):
        serving.bind_remote_policy(first)


def test_remote_model_mismatch_fails_before_rollouts(service_options, tmp_path, monkeypatch):
    from robocasa_eval.adapters import http
    metadata = serving.service_metadata(serving.build_server_config(service_options))
    metadata["model_id"] = "wrong-model"
    monkeypatch.setattr(http, "fetch_metadata", lambda *a, **kw: metadata)
    with pytest.raises(ValueError, match="does not match"):
        serving.bind_remote_policy(client_config(tmp_path))


def test_worker_aliases_and_cli_overrides_are_consistent(tmp_path):
    from robocasa_eval.cli import parser, read_options
    config = client_config(tmp_path)
    assert config["total_gpus"] == config["gpus_per_node"] == 16
    assert config["envs_per_gpu"] == 4 and config["device_prefix"] == "cpu"
    file = tmp_path / "client.json"
    file.write_text(json.dumps({"num_workers": 16, "envs_per_worker": 4}))
    args = parser().parse_args(["plan", "--config", str(file), "--num-workers", "1", "--envs-per-worker", "2"])
    options = read_options(args)
    assert options == {"total_gpus": 1, "envs_per_gpu": 2}
    with pytest.raises(ValueError, match="Conflicting"):
        configuration.build_config({"total_gpus": 1, "num_workers": 16}, environment={})


def test_serve_cli_does_not_import_simulator_or_need_decoder(service_options, tmp_path, monkeypatch, capsys):
    from robocasa_eval import cli, serving_http
    config_path = tmp_path / "service.json"
    config_path.write_text(json.dumps(service_options))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.gettempdir())
    calls = []
    def fake_serve(config, metadata):
        assert "sim_python" not in config and "decoder_path" not in config
        assert Path.cwd().is_relative_to(Path(config["workspace"]))
        assert (Path.cwd() / "metadata.json").is_file()
        assert config["port"] == 8123 and config["devices"] == ["cpu"]
        calls.append(metadata)
    monkeypatch.setattr(serving_http, "serve", fake_serve)
    monkeypatch.setattr(evaluation, "preflight", lambda *a: pytest.fail("serve must not preflight a simulator"))
    original = Path.cwd()
    assert cli.main(["serve", "--config", str(config_path), "--port", "8123", "--devices", "cpu"]) == 0
    assert len(calls) == 1 and Path.cwd() == original
    assert json.loads(capsys.readouterr().out)["status"] == "stopped"
