"""Exercise the real model-serving CLI and HTTP transport without GPU assets."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from robocasa_eval.adapters.http import HttpPolicy, fetch_metadata
from robocasa_eval.policy import ACTION_DIMS, EpisodeContext, STATE_DIMS


def test_serve_subprocess_predicts_and_stops_cleanly(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = tmp_path / "service.json"
    config.write_text(json.dumps(dict(
        workspace=str(workspace), devices=["cpu"], port=port,
        minimum_free_bytes=0, policy={
            "factory": "robocasa_eval.examples.custom_policy:create_policy",
            "id": "cli-interface-fixture", "kwargs": {"action_horizon": 3},
        },
    )))
    endpoint = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    console = workspace / "console.log"
    with console.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "robocasa_eval", "serve", "--config", str(config)],
            cwd=tmp_path, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                assert process.poll() is None, console.read_text()
                try:
                    metadata = fetch_metadata(endpoint, timeout=0.5)
                    break
                except RuntimeError:
                    assert time.monotonic() < deadline, console.read_text()
                    time.sleep(0.05)
            assert metadata["model_id"] == "cli-interface-fixture"
            assert metadata["replicas"] == 1
            client = HttpPolicy(
                endpoint=endpoint, workspace=workspace, device="cpu", seed=7,
                model_id=metadata["model_id"], fingerprint=metadata["fingerprint"],
            )
            context = EpisodeContext("cli-attempt-1", "PickPlace", 0, 7, 0, 0)
            try:
                client.reset(context)
                observation = {
                    "video": {"front": np.zeros((12, 16, 3), dtype=np.uint8)},
                    "state": {key: np.zeros(dim, dtype=np.float32) for key, dim in STATE_DIMS.items()},
                    "language": "Place the object on the counter.",
                }
                actions = client.predict_batch([observation], [context])
                assert len(actions) == 1
                assert {key: value.shape for key, value in actions[0].items()} == {
                    key: (3, dim) for key, dim in ACTION_DIMS.items()}
                client.end_episode(context)
            finally:
                client.close()
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 0, console.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    output = workspace / "model_service" / metadata["fingerprint"]
    assert json.loads((output / "metadata.json").read_text()) == metadata
    assert json.loads((output / "config.json").read_text())["workspace"] == str(workspace)
    log = (output / "server.log").read_text()
    assert '"event": "ready"' in log and "Model service stopped." in log
    with socket.socket() as connection:
        connection.settimeout(1)
        assert connection.connect_ex(("127.0.0.1", port)) != 0


def test_initialization_cause_is_logged_without_service_credential(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    module = tmp_path / "broken_policy.py"
    module.write_text(
        "import os\n"
        "def create_policy(**kwargs):\n"
        "    raise FileNotFoundError('Missing processor fixture; credential=' "
        "+ os.environ['POLICY_TEST_TOKEN'])\n"
    )
    token = "test-only-service-credential-123"
    config = tmp_path / "service.json"
    config.write_text(json.dumps(dict(
        workspace=str(workspace), devices=["cpu"], port=8000,
        minimum_free_bytes=0, auth_token_env="POLICY_TEST_TOKEN",
        policy={"factory": "broken_policy:create_policy", "id": "broken-cli-fixture"},
    )))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "robocasa_eval", "serve", "--config", str(config)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(tmp_path))),
                 PYTHONDONTWRITEBYTECODE="1", POLICY_TEST_TOKEN=token),
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 1
    logs = list((workspace / "model_service").glob("*/server.log"))
    assert len(logs) == 1, result.stderr
    diagnostic = logs[0].read_text()
    assert "FileNotFoundError: Missing processor fixture" in diagnostic
    assert "Policy initialization failed on cpu" in diagnostic
    assert "[redacted]" in diagnostic
    assert token not in diagnostic + result.stdout + result.stderr
