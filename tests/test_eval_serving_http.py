"""Real loopback tests for the model-owned single-port inference protocol."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
import http.client
import json
import signal
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from robocasa_eval.adapters.http import (
    FINGERPRINT_HEADER, HttpPolicy, PROTOCOL, decode_value, encode_value, fetch_metadata,
)
from robocasa_eval.policy import ACTION_DIMS, EpisodeContext
from robocasa_eval.serving_http import make_server


def metadata():
    return {"protocol": PROTOCOL, "model_id": "test-model-v1", "fingerprint": "a" * 64,
            "action_dims": dict(ACTION_DIMS), "replicas": 2}


def context(episode=0, rank=0):
    return EpisodeContext(f"test-run:{rank}:{episode}", "OpenDrawer", episode, 123 + episode, rank, episode % 2)


def observation():
    return {"video": {"ego_view": np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)},
            "language": "Open the drawer.", "state": {"waist": np.arange(3, dtype=np.float32)}}


class Pool:
    def __init__(self):
        self.active = {}
        self.lock = threading.Lock()
        self.closed = 0
        self.fail_predict = None
        self.calls = []

    def reset(self, context):
        from robocasa_eval.serving_pool import PolicyContextError
        with self.lock:
            if context.episode_id in self.active:
                raise PolicyContextError("duplicate with secret diagnostic")
            self.active[context.episode_id] = [context, 0]
            self.calls.append(("reset", context))

    def predict_batch(self, observations, contexts):
        from robocasa_eval.serving_pool import PolicyContextError
        if self.fail_predict:
            raise self.fail_predict
        with self.lock:
            result = []
            for obs, context in zip(observations, contexts):
                if context.episode_id not in self.active or self.active[context.episode_id][0] != context:
                    raise PolicyContextError("unknown secret context")
                assert obs["video"]["ego_view"].dtype == np.uint8
                self.active[context.episode_id][1] += 1
                offset = context.rank * 100 + self.active[context.episode_id][1]
                result.append({key: np.full((context.episode + 1, dim), offset, dtype=np.float32)
                               for key, dim in ACTION_DIMS.items()})
            self.calls.append(("predict", list(contexts)))
            return result

    def end_episode(self, context):
        from robocasa_eval.serving_pool import PolicyContextError
        with self.lock:
            if context.episode_id not in self.active or self.active[context.episode_id][0] != context:
                raise PolicyContextError("unknown secret context")
            del self.active[context.episode_id]
            self.calls.append(("end", context))

    def close(self):
        self.closed += 1
        self.active.clear()


@contextmanager
def running_server(**options):
    pool = Pool()
    server = make_server(pool, metadata(), port=0, **options)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    endpoint = "http://127.0.0.1:" + str(server.server_address[1])
    try:
        yield server, pool, endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
        assert pool.closed == 1


def request(endpoint, path, payload=None, *, raw=None, headers=None, method="POST"):
    if payload is not None:
        raw = json.dumps(encode_value(payload)).encode()
    req = urllib.request.Request(endpoint + path, data=raw, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, decode_value(json.loads(response.read()))


def test_real_port_metadata_readiness_and_model_identity_handshake(tmp_path):
    with running_server() as (_, pool, endpoint):
        assert fetch_metadata(endpoint, timeout=3) == metadata()
        assert request(endpoint, "/health", method="GET") == (200, {**metadata(), "ready": True})
        assert pool.calls == []
        with pytest.raises(ValueError, match="model_id"):
            HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                       model_id="another-model", timeout=3)
        with pytest.raises(ValueError, match="fingerprint"):
            HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                       fingerprint="b" * 64, timeout=3)
        policy = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                            model_id=metadata()["model_id"], fingerprint=metadata()["fingerprint"], timeout=3)
        assert policy.metadata == metadata()
        policy.reset(context())
        result = policy.predict_batch([observation()], [context()])
        assert result[0]["motion_token"].shape == (1, 64)
        policy.end_episode(context())
        policy.close()
        assert not pool.active


def test_multiple_clients_preserve_episode_history_with_reordered_batches(tmp_path):
    with running_server() as (_, pool, endpoint):
        def exercise(rank):
            policy = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=rank,
                                model_id=metadata()["model_id"], timeout=3)
            first, second, third = context(0, rank), context(1, rank), context(2, rank)
            policy.reset(first)
            policy.reset(second)
            result = policy.predict_batch([observation(), observation()], [second, first])
            assert [item["motion_token"].shape[0] for item in result] == [2, 1]
            policy.end_episode(first)
            policy.reset(third)
            result = policy.predict_batch([observation(), observation()], [third, second])
            assert [float(item["motion_token"][0, 0]) for item in result] == [rank * 100 + 1, rank * 100 + 2]
            policy.close()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(exercise, range(4)))
        assert not pool.active


def test_fingerprint_prevents_requests_after_service_model_changes(tmp_path):
    with running_server() as (server, pool, endpoint):
        client = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                            model_id=metadata()["model_id"], timeout=3)
        client.reset(context())
        server.metadata["fingerprint"] = "b" * 64
        with pytest.raises(RuntimeError, match="status 409"):
            client.predict_batch([observation()], [context()])
        assert [call[0] for call in pool.calls] == ["reset"]
        with pytest.raises(RuntimeError, match="status 409"):
            client.close()
        assert client._closed


def test_client_verifies_response_fingerprint_even_when_server_ignores_header(tmp_path):
    with running_server() as (server, pool, endpoint):
        client = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                            model_id=metadata()["model_id"], timeout=3)
        client.reset(context())
        del client._headers[FINGERPRINT_HEADER]
        server.metadata["fingerprint"] = "b" * 64
        with pytest.raises(ValueError, match="response fingerprint"):
            client.predict_batch([observation()], [context()])
        with pytest.raises(ValueError, match="response fingerprint"):
            client.close()


def test_authenticated_metadata_and_actions_do_not_expose_credentials(tmp_path, monkeypatch):
    token = "secret-unit-test-credential"
    monkeypatch.setenv("TEST_MODEL_TOKEN", token)
    with running_server(auth_token_env="TEST_MODEL_TOKEN") as (_, pool, endpoint):
        status, response = request(endpoint, "/metadata", method="GET")
        assert status == 401
        assert token not in json.dumps(response)
        with pytest.raises(RuntimeError, match="status 401"):
            fetch_metadata(endpoint)
        assert fetch_metadata(endpoint, auth_token_env="TEST_MODEL_TOKEN") == metadata()
        policy = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                            model_id=metadata()["model_id"], auth_token_env="TEST_MODEL_TOKEN", timeout=3)
        policy.reset(context())
        policy.close()
        assert not pool.active


@pytest.mark.parametrize("changes", [
    {"episode_id": ""}, {"task": " "}, {"episode": True}, {"episode": -1},
    {"seed": 0.5}, {"rank": -1}, {"slot": "0"}, {"extra": "field"},
])
def test_strict_context_rejected_before_model_is_called(changes):
    with running_server() as (_, pool, endpoint):
        status, response = request(endpoint, "/reset", {"protocol": PROTOCOL, "context": {**asdict(context()), **changes}})
        assert status == 400
        assert "error" in response
        assert pool.calls == []


@pytest.mark.parametrize("raw", [
    b"not json", b"{", b'{"protocol":"bad","context":{}}',
    b'{"protocol":"robocasa-g1-v1","protocol":"robocasa-g1-v1","context":{}}',
    b'{"protocol":"robocasa-g1-v1","context":NaN}',
])
def test_invalid_request_formats_return_json_400(raw):
    with running_server() as (_, pool, endpoint):
        status, response = request(endpoint, "/reset", raw=raw)
        assert status == 400
        assert response["protocol"] == PROTOCOL
        assert pool.calls == []


def test_size_content_type_and_unknown_routes_are_rejected():
    with running_server(max_request_bytes=32) as (_, pool, endpoint):
        assert request(endpoint, "/reset", raw=b" " * 33)[0] == 413
        assert request(endpoint, "/reset", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 400
        assert request(endpoint, "/missing", raw=b"{}")[0] == 404
        assert request(endpoint, "/missing", method="GET")[0] == 404
        assert pool.calls == []


def test_body_read_timeout_returns_408_without_waiting_for_client():
    with running_server(read_timeout=0.05) as (server, pool, _):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        try:
            conn.putrequest("POST", "/reset")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", "100")
            conn.endheaders()
            response = conn.getresponse()
            assert response.status == 408
            assert json.loads(response.read())["error"] == "Request body read timed out."
            assert pool.calls == []
        finally:
            conn.close()


def test_context_conflicts_and_model_errors_have_distinct_statuses():
    with running_server() as (_, pool, endpoint):
        reset = {"protocol": PROTOCOL, "context": asdict(context())}
        assert request(endpoint, "/reset", reset)[0] == 200
        status, response = request(endpoint, "/reset", reset)
        assert status == 409 and "secret" not in str(response)
        predict = {"protocol": PROTOCOL, "observations": [observation()], "contexts": [asdict(replace(context(), seed=900))]}
        assert request(endpoint, "/predict", predict)[0] == 409
        predict["contexts"] = [asdict(context())]
        for error, expected_status in [(RuntimeError("private model traceback"), 500),
                                       (ValueError("private tensor diagnostic"), 500),
                                       (TimeoutError("private timeout details"), 504)]:
            pool.fail_predict = error
            status, response = request(endpoint, "/predict", predict)
            assert status == expected_status
            assert "private" not in str(response)


@pytest.mark.parametrize("error_type,expected_status", [
    (ValueError, 500), (RuntimeError, 500), (TimeoutError, 504),
])
def test_model_failure_traceback_is_logged_without_request_data_or_credentials(
    error_type, expected_status, monkeypatch, capsys,
):
    token = "diagnostic-test-secret-token"
    monkeypatch.setenv("TEST_MODEL_TOKEN", token)
    headers = {"Authorization": "Bearer " + token, "X-Private-Test-Header": "private-header-value"}
    with running_server(auth_token_env="TEST_MODEL_TOKEN") as (_, pool, endpoint):
        request(endpoint, "/reset", {"protocol": PROTOCOL, "context": asdict(context())}, headers=headers)
        obs = observation()
        obs["language"] = "private-observation-language"
        pool.fail_predict = error_type("diagnostic-model-failure: " + token)
        status, response = request(endpoint, "/predict", {
            "protocol": PROTOCOL, "observations": [obs], "contexts": [asdict(context())],
        }, headers=headers)
        assert status == expected_status
        assert "diagnostic-model-failure" not in str(response)
        diagnostic = capsys.readouterr().err
        assert "Model operation /predict failed:" in diagnostic
        assert "Traceback (most recent call last)" in diagnostic
        assert error_type.__name__ + ": diagnostic-model-failure: [redacted]" in diagnostic
        for private_value in (token, "private-header-value", "private-observation-language",
                              encode_value(obs["video"]["ego_view"])["data"]):
            assert private_value not in diagnostic


def test_nonfinite_or_malformed_model_actions_never_reach_clients():
    with running_server() as (_, pool, endpoint):
        request(endpoint, "/reset", {"protocol": PROTOCOL, "context": asdict(context())})
        def invalid_actions(*args):
            return [{key: np.full((1, dim), np.nan) for key, dim in ACTION_DIMS.items()}]
        pool.predict_batch = invalid_actions
        status, response = request(endpoint, "/predict", {"protocol": PROTOCOL, "observations": [observation()],
                                                           "contexts": [asdict(context())]})
        assert status == 500
        assert "actions" not in response


@pytest.mark.parametrize("update", [
    {"protocol": "other"}, {"model_id": ""}, {"fingerprint": "a\nb"},
    {"replicas": True}, {"replicas": 0}, {"action_dims": {"action": 78}},
])
def test_server_rejects_invalid_metadata_before_binding(update):
    with pytest.raises(ValueError):
        make_server(Pool(), {**metadata(), **update}, port=0)


def test_make_server_copies_metadata_and_closes_pool_once():
    pool = Pool()
    declared = metadata()
    server = make_server(pool, declared, port=0)
    declared["model_id"] = "changed"
    assert server.metadata["model_id"] == "test-model-v1"
    server.server_close()
    server.server_close()
    assert pool.closed == 1


def test_real_pool_and_real_port_route_multiple_clients_to_model_replicas(tmp_path):
    from robocasa_eval.serving_pool import PolicyPool

    pool = PolicyPool({
        "workspace": str(tmp_path), "devices": ["cpu:0", "cpu:1"],
        "policy": {"factory": "robocasa_eval.examples.custom_policy:create_policy",
                   "id": "test-model-v1", "kwargs": {"action_horizon": 20}, "artifacts": []},
        "batch_wait_ms": 10, "max_batch_size": 4, "request_timeout_seconds": 3,
    })
    server = make_server(pool, metadata(), port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    endpoint = "http://127.0.0.1:" + str(server.server_address[1])
    try:
        def exercise(rank):
            client = HttpPolicy(endpoint=endpoint, workspace=tmp_path, device="cpu", seed=0,
                                model_id="test-model-v1", timeout=3)
            try:
                client.reset(context(rank=rank))
                actions = client.predict_batch([observation()], [context(rank=rank)])
                assert actions[0]["motion_token"].shape == (20, 64)
            finally:
                client.close()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(exercise, range(4)))
        assert pool.active_episode_count == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert not thread.is_alive()
    assert all(not replica.thread.is_alive() for replica in pool._replicas)


def test_serve_sigterm_closes_pool_and_restores_signal_handler(monkeypatch):
    from robocasa_eval import evaluation, gpu_worker, serving_http, serving_pool

    pool = Pool()
    original_handler = signal.getsignal(signal.SIGTERM)

    class Server:
        server_address = ("127.0.0.1", 8000)
        metadata = metadata()

        def serve_forever(self, **kwargs):
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

        def server_close(self):
            pool.close()

    monkeypatch.setattr(serving_pool, "PolicyPool", lambda config: pool)
    monkeypatch.setattr(serving_http, "make_server", lambda *args, **kwargs: Server())
    monkeypatch.setattr(evaluation, "disk_preflight", lambda config: {})
    monkeypatch.setattr(gpu_worker, "DiskQuotaMonitor", lambda config: None)
    serving_http.serve({}, metadata())
    assert pool.closed == 1
    assert signal.getsignal(signal.SIGTERM) == original_handler


def test_health_returns_unready_when_model_pool_stops():
    with running_server() as (_, pool, endpoint):
        pool.ready = False
        status, response = request(endpoint, "/health", method="GET")
        assert status == 503
        assert response["ready"] is False


def test_serve_quota_failure_closes_models_and_background_monitor(monkeypatch):
    from robocasa_eval import evaluation, gpu_worker, serving_http, serving_pool

    pool = Pool()

    class Monitor:
        closed = False

        def poll(self):
            raise ValueError("Workspace exceeds configured quota")

        def close(self):
            self.closed = True

    monitor = Monitor()
    monkeypatch.setattr(serving_pool, "PolicyPool", lambda config: pool)
    monkeypatch.setattr(evaluation, "disk_preflight", lambda config: {})
    monkeypatch.setattr(gpu_worker, "DiskQuotaMonitor", lambda config: monitor)
    with pytest.raises(ValueError, match="quota"):
        serving_http.serve({"port": 0}, metadata())
    assert monitor.closed
    assert pool.closed == 1


def test_serve_rechecks_quota_after_loading_before_opening_port(monkeypatch):
    from robocasa_eval import evaluation, serving_http, serving_pool

    pool = Pool()
    monkeypatch.setattr(serving_pool, "PolicyPool", lambda config: pool)

    def failed_quota(config):
        raise ValueError("Workspace exceeds configured quota")

    def must_not_bind(*args, **kwargs):
        pytest.fail("Quota failure must be detected before a listening port opens")

    monkeypatch.setattr(evaluation, "disk_preflight", failed_quota)
    monkeypatch.setattr(serving_http, "make_server", must_not_bind)
    with pytest.raises(ValueError, match="quota"):
        serving_http.serve({}, metadata())
    assert pool.closed == 1
