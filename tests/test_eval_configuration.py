"""Portable configuration and CLI checks without real model or simulator assets."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from robocasa_eval import configuration as conf, evaluation


@pytest.fixture
def options(tmp_path):
    return {"workspace": str(tmp_path / "workspace"), "envs_per_gpu": 2}


def build(options, **kwargs):
    return conf.build_config(options, environment={}, **kwargs)[0]


def test_plan_needs_only_stdlib_and_creates_no_files(options, tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "not-created"
    code = '''
import importlib.abc, sys
class DenyFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy', 'torch', 'mujoco', 'robocasa'}:
            raise AssertionError('Plan imported ' + fullname)
sys.meta_path.insert(0, DenyFrameworks())
from robocasa_eval.cli import main
raise SystemExit(main(['plan', '--workspace', sys.argv[1]]))
'''
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    for key in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE", "NODE_RANK", "NNODES", "GPUS_PER_NODE"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, "-B", "-S", "-c", code, str(destination)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["total_episodes"] == 1500
    assert sum(report["rank_episode_counts"].values()) == 1500
    assert report["config"]["policy_identity"]["verified"] is False
    assert report["config"]["policy"] == {
        "factory": "http", "id": None,
        "kwargs": {"endpoint": "http://127.0.0.1:8000"}, "artifacts": [],
    }
    assert report["config"]["device_prefix"] == "cpu"
    assert not destination.exists()


@pytest.mark.parametrize("update", [
    {"unknown": 2}, {"total_gpus": True}, {"episodes_per_task": 0},
    {"envs_per_gpu": 33}, {"envs_per_gpu": "2"}, {"run_name": "../escape"},
    {"atomic_seconds": float("nan")}, {"composite_seconds": float("inf")},
    {"atomic_seconds": 0.015}, {"camera_names": [[]]}, {"camera_names": []},
    {"camera_names": ["ego_view", "ego_view"]}, {"camera_names": ["external"]},
    {"renderer": []}, {"layout_ids": None}, {"style_ids": True},
    {"camera_width": 4096, "camera_height": 4096, "camera_names": ["ego_view"]},
    {"policy": {"factory": ""}},
    {"policy": {"id": ""}}, {"policy": {"id": "   "}}, {"policy": {"id": 1}},
    {"policy": {"factory": "not_a_factory", "id": "v1"}},
    {"policy": {"factory": "http", "id": "v1", "kwargs": {"device": "cuda:2"}}},
])
def test_reject_invalid_options(options, update):
    with pytest.raises(ValueError):
        build(dict(options, **update))


def test_options_are_not_mutated_and_custom_factory_kwargs_stay_opaque(options):
    options["policy"] = {
        "factory": "some_package:create_policy", "kwargs": {"checkpoint": "./relative", "repo": "some/repo"},
        "artifacts": ["./declared-artifact"],
    }
    before = copy.deepcopy(options)
    config = build(options)
    assert options == before
    assert config["policy"]["kwargs"] == before["policy"]["kwargs"]
    assert Path(config["policy"]["artifacts"][0]).is_absolute()
    assert config["policy"]["id"] is None


@pytest.mark.parametrize("policy", [{}, {"kwargs": {"endpoint": "http://127.0.0.1:9000"}},
    {"factory": "http"}, {"factory": "robocasa_eval.adapters.http:create_policy"}])
def test_service_configuration_does_not_require_model_type_or_revision(options, policy):
    config = build(dict(options, policy=policy))
    assert config["policy"]["id"] is None
    assert config["policy_identity"]["factory"] == "robocasa_eval.adapters.http:create_policy"
    assert config["policy"]["kwargs"]["endpoint"] == policy.get("kwargs", {}).get(
        "endpoint", "http://127.0.0.1:8000")


def test_simulator_interpreter_retains_virtualenv_packages(options, tmp_path):
    import venv
    sim_env = tmp_path / "sim-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(sim_env)
    interpreter = sim_env / "bin" / "python"
    assert interpreter.is_symlink()
    site = sim_env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site / "sim_env_marker.py").write_text("VALUE = 'simulator-only-package'\n")
    config = build(dict(options, sim_python=str(interpreter)))
    environment = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(
        [config["sim_python"], "-B", "-c",
         "import json, sys, sim_env_marker; print(json.dumps([sys.prefix, sim_env_marker.VALUE]))"],
        env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(sim_env), "simulator-only-package"]
    assert config["sim_python"] == str(interpreter)


def test_declared_model_and_bytes_change_fingerprint(options, tmp_path):
    model = tmp_path / "weights.bin"
    decoder = tmp_path / "decoder.onnx"
    model.write_bytes(b"first-weights")
    decoder.write_bytes(b"decoder-fixture")
    options.update(decoder_path=str(decoder), sim_python=sys.executable,
                   policy={"artifacts": [str(model)]})
    first = build(options, verify_files=True)
    assert first["policy_identity"]["artifacts"][0]["sha256"] == conf.sha256_file(model)
    assert first["policy_identity"]["id"] is None
    assert "runtime" not in first["policy_identity"]
    model.write_bytes(b"other-weights")
    changed = build(options, verify_files=True)
    assert first["identity"] != changed["identity"]
    assert conf.gate_protocol(first) != conf.gate_protocol(changed)
    model.write_bytes(b"first-weights")
    assert build(options, verify_files=True)["identity"] == first["identity"]
    decoder.write_bytes(b"changed-decoder")
    assert build(options, verify_files=True)["identity"] != first["identity"]


def test_custom_manifest_uses_declared_groups(options, tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"tasks": [dict(name="ExampleTask", category="atomic", split="unseen")]}))
    options["task_manifest"] = str(manifest)
    config = build(options)
    report = evaluation.summarize(config, write=False)
    assert config["benchmark"] == "custom_tasks"
    assert report["groups"]["unseen"]["planned"] == 30
    assert report["groups"]["seen"]["planned"] == 0
    assert report["groups"]["seen"]["success_rate_planned"] is None


@pytest.mark.parametrize("data", [[], {}, {"tasks": []},
    {"tasks": [{"name": None, "category": "atomic", "split": "seen"}]},
    {"tasks": [{"name": "Foo", "category": [], "split": "seen"}]},
    {"tasks": [dict(name="Foo", category="atomic", split="seen")] * 2}])
def test_custom_manifest_rejects_malformed_tasks(data, tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        conf.load_tasks(manifest)


def test_cache_paths_and_temporary_files_stay_in_workspace(options, monkeypatch):
    root = Path(options["workspace"])
    hub = root / "weights/hub"
    hub.mkdir(parents=True)
    # Capture global environment and tempfile cache changes for pytest isolation.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.gettempdir())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    config = build(options)
    evaluation.prepare_workspace(config)
    assert os.environ["HF_HUB_CACHE"] == str(hub)
    for key in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR"):
        assert Path(os.environ[key]).is_relative_to(root)
    with tempfile.TemporaryFile() as stream:
        stream.write(b"workspace scratch")
    assert Path(tempfile.gettempdir()) == root / "tmp"


def test_workspace_symlink_escape_is_rejected(options, tmp_path, monkeypatch):
    root = Path(options["workspace"])
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (root / "tmp").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    with pytest.raises(ValueError, match="Writable path"):
        evaluation.prepare_workspace(build(options))


def test_same_gate_for_single_device_and_sixteen_device_run(options):
    single = build(dict(options, total_gpus=1, run_name="single"))
    sixteen = build(dict(options, total_gpus=16, run_name="formal"))
    assert single["identity"] != sixteen["identity"]
    assert conf.gate_protocol(single) == conf.gate_protocol(sixteen)


def test_resume_report_does_not_require_model_files(options, tmp_path, capsys, monkeypatch):
    from robocasa_eval.cli import main
    config = build(options)
    evaluation.install_manifest(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(options, task_manifest=str(tmp_path / "removed-tasks.json"),
        policy={"factory": "package_missing:factory", "id": "missing"})))
    for key in ("NNODES", "WORLD_SIZE", "GPUS_PER_NODE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    assert main(["summarize", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["groups"]["all"]["pending"] == 1500
