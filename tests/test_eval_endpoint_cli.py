"""An inference URL is sufficient to select the shared HTTP evaluation client."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from robocasa_eval.cli import parser, read_options


def test_endpoint_cli_plans_without_model_name_framework_or_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "not-created"
    script = '''
import importlib.abc, sys
class DenyFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy', 'torch', 'mujoco', 'robocasa', 'transformers'}:
            raise AssertionError('Plan imported ' + fullname)
sys.meta_path.insert(0, DenyFrameworks())
from robocasa_eval.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    for key in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE", "NODE_RANK", "NNODES", "GPUS_PER_NODE"):
        environment.pop(key, None)
    command = [sys.executable, "-B", "-S", "-c", script,
               "plan", "--endpoint", "http://127.0.0.1:1/inference",
               "--sim-python", str(tmp_path / "sim/bin/python"),
               "--decoder-path", str(tmp_path / "decoder.onnx"),
               "--workspace", str(destination), "--num-workers", "16", "--envs-per-worker", "4"]
    result = subprocess.run(command, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["total_episodes"] == 1500
    assert report["max_concurrent_environments"] == 64
    config = report["config"]
    assert config["policy"]["factory"] == "http"
    assert config["policy"]["id"] is None
    assert config["policy"]["kwargs"] == {"endpoint": "http://127.0.0.1:1/inference"}
    assert config["device_prefix"] == "cpu"
    assert config["atomic_seconds"] == 40 and config["composite_seconds"] == 180
    assert not destination.exists()


def test_endpoint_override_preserves_client_options_and_source_file(tmp_path):
    source = {"policy": {"kwargs": {"endpoint": "http://old:8000", "timeout": 23,
                                   "auth_token_env": "INFERENCE_TOKEN"}}, "num_workers": 8}
    path = tmp_path / "client.json"
    path.write_text(json.dumps(source))
    args = parser().parse_args(["check", "--config", str(path), "--endpoint", "http://new:8000"])
    options = read_options(args)
    assert options["policy"]["kwargs"] == dict(source["policy"]["kwargs"], endpoint="http://new:8000")
    assert options["policy"]["factory"] == "http" and "id" not in options["policy"]
    assert options["num_workers"] == 8
    assert json.loads(path.read_text()) == source


def test_endpoint_does_not_silently_convert_a_local_adapter(tmp_path):
    path = tmp_path / "local.json"
    path.write_text(json.dumps({"policy": {"factory": "custom:create_policy"}}))
    args = parser().parse_args(["run", "--config", str(path), "--endpoint", "http://127.0.0.1:8000"])
    with pytest.raises(ValueError, match="HTTP evaluation client"):
        read_options(args)


@pytest.mark.parametrize("factory", [[], {}, 1, None])
def test_endpoint_reports_invalid_factory_as_configuration_error(tmp_path, capsys, factory):
    from robocasa_eval.cli import main

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"policy": {"factory": factory}}))
    assert main(["plan", "--config", str(path), "--endpoint", "http://127.0.0.1:8000"]) == 1
    captured = capsys.readouterr()
    assert "HTTP evaluation client configuration" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("option,value", [("--endpoint", "http://127.0.0.1:8000"),
                                          ("--sim-python", "/example/python"),
                                          ("--decoder-path", "/example/decoder.onnx")])
def test_client_options_rejected_by_serve(option, value):
    with pytest.raises(ValueError, match="evaluation client"):
        read_options(parser().parse_args(["serve", option, value]))
