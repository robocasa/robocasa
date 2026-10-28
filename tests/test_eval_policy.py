"""CPU-only regression tests for public policy boundaries and HTTP transport."""

from dataclasses import FrozenInstanceError, replace
import json
import sys
from types import ModuleType, SimpleNamespace
import urllib.error

import numpy as np
import pytest

from robocasa_eval.policy import (
    ACTION_DIMS, STATE_DIMS, BasePolicy, EpisodeContext, load_policy,
    resolve_factory, validate_action_batch,
)
from robocasa_eval.adapters.http import PROTOCOL, HttpPolicy, decode_value, encode_value
from robocasa_eval.examples.custom_policy import ExamplePolicy


@pytest.fixture
def service_metadata(monkeypatch):
    metadata = {"protocol": PROTOCOL, "fingerprint": "test-fingerprint", "action_dims": ACTION_DIMS}
    monkeypatch.setattr("robocasa_eval.adapters.http.fetch_metadata", lambda *args, **kwargs: metadata)
    return metadata


def context(episode=0, slot=0):
    return EpisodeContext(f"run:0:Task:{episode}", "Task", episode, 500 + episode, 0, slot)


def action(horizon=3, offset=0):
    return {
        key: np.arange(horizon * dim, dtype=np.float64).reshape(horizon, dim) + offset
        for key, dim in ACTION_DIMS.items()
    }


def observation():
    return {
        "video": {"ego_view": np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)},
        "state": {key: np.arange(dim, dtype=np.float32) for key, dim in STATE_DIMS.items()},
        "language": "Open the drawer.",
    }


def test_context_is_immutable_and_identity_survives_batch_reordering(tmp_path):
    first, second, replacement = context(0, 0), context(1, 1), context(2, 0)
    with pytest.raises(FrozenInstanceError):
        first.slot = 2
    policy = ExamplePolicy(device="cpu", seed=9, workspace=tmp_path)
    policy.reset(first)
    policy.reset(second)
    policy.predict_batch([{}, {}], [first, second])
    policy.predict_batch([{}], [second])
    policy.end_episode(first)
    policy.reset(replacement)
    policy.predict_batch([{}, {}], [second, replacement])
    assert policy._episodes[second.episode_id]["inferences"] == 3
    assert policy._episodes[replacement.episode_id]["inferences"] == 1
    assert first.episode_id not in policy._episodes
    with pytest.raises(ValueError, match="active episode"):
        policy.predict_batch([{}], [first])
    with pytest.raises(ValueError, match="active episode"):
        policy.predict_batch([{}], [replace(second, seed=123)])
    policy.close()
    assert policy._episodes == {}


def test_actions_accept_variable_horizons_and_are_copied_before_execution():
    inputs = [action(7), action(1), action(5)]
    result = validate_action_batch(inputs, 3, 4, remaining_steps=[2, 8, 9])
    assert [item["motion_token"].shape[0] for item in result] == [2, 1, 4]
    for original, converted in zip(inputs, result):
        for key in ACTION_DIMS:
            assert converted[key].dtype == np.float32
            assert converted[key].flags.c_contiguous
            assert not np.shares_memory(original[key], converted[key])
            np.testing.assert_array_equal(converted[key], original[key][:len(converted[key])])
    result[0]["motion_token"][0, 0] = -100
    assert inputs[0]["motion_token"][0, 0] == 0
    assert validate_action_batch([], 0, 4) == []


@pytest.mark.parametrize("mutate", [
    lambda a: a.update({"extra": np.zeros((3, 1))}),
    lambda a: a.pop("left_hand_joints"),
    lambda a: a.update(motion_token=np.zeros(64)),
    lambda a: a.update(motion_token=np.zeros((3, 63))),
    lambda a: a.update(left_hand_joints=np.zeros((2, 7))),
    lambda a: a.update({key: np.zeros((0, dim)) for key, dim in ACTION_DIMS.items()}),
    lambda a: a["motion_token"].__setitem__((2, 0), np.nan),
    lambda a: a["right_hand_joints"].__setitem__((2, 0), np.inf),
    lambda a: a.update(motion_token=np.full((3, 64), 1e100)),
    lambda a: a.update(motion_token=np.full((3, 64), "1")),
    lambda a: a.update(motion_token=np.ones((3, 64), dtype=np.complex64)),
    lambda a: a.update(motion_token=np.ones((3, 64), dtype=bool)),
])
def test_invalid_actions_including_unexecuted_tail_are_rejected(mutate):
    values = action()
    mutate(values)
    with pytest.raises(ValueError):
        validate_action_batch([values], 1, 1)


@pytest.mark.parametrize("args", [
    ([action()], 2, 1), (action(), 1, 1), ([action()], 1, 0),
    ([action()], True, 1), ([action()], 1, True),
    ([action()], 1, 1, []), ([action()], 1, 1, [0]),
    ([action()], 1, 1, [True]), ([action()], 1, 1, [1.5]),
])
def test_action_batch_dimensions_and_execution_limits_are_strict(args):
    with pytest.raises(ValueError):
        validate_action_batch(*args)


def test_factory_receives_explicit_workspace_and_only_its_kwargs(tmp_path, monkeypatch):
    module = ModuleType("test_custom_vla_factory")
    captured = {}

    class Policy(BasePolicy):
        def predict_batch(self, observations, contexts):
            return [action(1) for _ in observations]

    def factory(**kwargs):
        captured.update(kwargs)
        return Policy()

    module.create = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    policy = load_policy({
        "workspace": str(tmp_path),
        "policy": {"factory": f"{module.__name__}:create", "kwargs": {"temperature": 0.1},
                   "id": "model-v1", "artifacts": []},
    }, device="cuda:2", seed=13)
    assert captured == {"device": "cuda:2", "seed": 13, "workspace": tmp_path.resolve(), "temperature": 0.1}
    assert policy.reset(context()) is None
    assert policy.end_episode(context()) is None
    assert policy.close() is None


@pytest.mark.parametrize("policy", [
    {"factory": None}, {"factory": []},
    {"factory": "missing-colon"}, {"factory": "abc:def:ghi"},
    {"factory": "x:y", "kwargs": {"workspace": "/tmp"}},
    {"factory": "x:y", "kwargs": {"device": "cuda:3"}},
    {"factory": "x:y", "kwargs": {"seed": 0}},
    {"factory": "x:y", "kwagrs": {}},
])
def test_factory_rejects_ambiguous_or_reserved_config(tmp_path, policy):
    with pytest.raises(ValueError):
        load_policy({"workspace": str(tmp_path), "policy": policy}, device="cpu", seed=0)


def test_http_factory_is_lazy_and_does_not_import_gpu_runtime(tmp_path, monkeypatch, service_metadata):
    import robocasa_eval.policy as policy_module
    imported = []
    original = policy_module.importlib.import_module

    def guarded_import(name, *args, **kwargs):
        imported.append(name)
        assert not name.startswith(("torch", "mujoco", "robocasa."))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(policy_module.importlib, "import_module", guarded_import)
    policy = load_policy({"workspace": str(tmp_path), "policy": {
        "factory": "http", "kwargs": {"endpoint": "http://127.0.0.1:9000/policy"}
    }}, device="cpu", seed=0)
    assert isinstance(policy, HttpPolicy)
    assert imported == ["robocasa_eval.adapters.http"]
    policy.close()


def test_policy_factory_defaults_to_http_without_model_name(tmp_path, service_metadata):
    policy = load_policy({"workspace": str(tmp_path), "policy": {}}, device="cpu", seed=0)
    assert isinstance(policy, HttpPolicy)
    assert policy._fingerprint == service_metadata["fingerprint"]
    policy.close()


def test_resolve_builtin_factories_does_not_load_models(monkeypatch):
    import robocasa_eval.policy as policy_module
    original = policy_module.importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("torch", "mujoco", "robocasa."))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(policy_module.importlib, "import_module", guarded_import)
    assert resolve_factory("http").__module__ == "robocasa_eval.adapters.http"


def test_http_array_roundtrip_preserves_images_noncontiguous_arrays_and_endianness():
    source = {
        "image": observation()["video"]["ego_view"][:, ::2],
        "state": np.arange(24, dtype=">f4").reshape(4, 6).T,
        "scalar": np.int64(42), "language": "移动杯子", "bool": np.array([True]),
    }
    encoded = encode_value(source)
    decoded = decode_value(json.loads(json.dumps(encoded)))
    assert encoded["image"]["dtype"] == "|u1"
    assert encoded["state"]["dtype"] == ">f4"
    for key in ("image", "state", "bool"):
        np.testing.assert_array_equal(decoded[key], source[key])
        assert decoded[key].dtype == source[key].dtype
        assert not np.shares_memory(decoded[key], source[key])
    assert decoded["language"] == "移动杯子"
    assert decoded["scalar"] == 42


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(shape=[999, 64]),
    lambda a: a.update(shape=[True, 64]),
    lambda a: a.update(shape=[-1, 64]),
    lambda a: a.update(shape=[2**40, 64]),
    lambda a: a.update(dtype="object"),
    lambda a: a.update(dtype="complex64"),
    lambda a: a.update(data="!invalid base64!"),
    lambda a: a.update(__ndarray__=False),
    lambda a: a.update(extra=True),
])
def test_http_array_decoder_rejects_malformed_payloads(mutate):
    encoded = encode_value(action()["motion_token"])
    mutate(encoded)
    with pytest.raises(ValueError):
        decode_value(encoded)


class FakeResponse:
    status = 200

    def __init__(self, value=None, raw=None):
        if isinstance(value, dict) and value.get("protocol") == PROTOCOL:
            value = dict(value, fingerprint="test-fingerprint")
        self.raw = raw if raw is not None else json.dumps(encode_value(value)).encode()

    def read(self, size):
        return self.raw[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class StatefulServer:
    """Protocol peer verifying that reordered batches do not mix histories."""

    def __init__(self):
        self.active = {}
        self.requests = []

    def open(self, request, timeout):
        payload = decode_value(json.loads(request.data))
        self.requests.append(payload)
        assert request.method == "POST"
        assert timeout == 3
        assert payload["protocol"] == PROTOCOL
        if request.full_url.endswith("/reset"):
            context_data = payload["context"]
            episode_id = context_data["episode_id"]
            assert episode_id not in self.active
            self.active[episode_id] = [context_data, 0]
            return FakeResponse({"protocol": PROTOCOL, "ok": True})
        if request.full_url.endswith("/end_episode"):
            context_data = payload["context"]
            assert self.active[context_data["episode_id"]][0] == context_data
            del self.active[context_data["episode_id"]]
            return FakeResponse({"protocol": PROTOCOL, "ok": True})
        assert request.full_url.endswith("/predict")
        results = []
        for obs, context_data in zip(payload["observations"], payload["contexts"]):
            state = self.active[context_data["episode_id"]]
            assert state[0] == context_data
            assert obs["video"]["ego_view"].dtype == np.uint8
            state[1] += 1
            results.append(action(context_data["episode"] + 1, offset=state[1]))
        return FakeResponse({"protocol": PROTOCOL, "actions": results})


def test_http_lifecycle_retains_context_when_batches_reorder_and_slot_is_reused(tmp_path, service_metadata):
    client = HttpPolicy(endpoint="http://localhost/prefix", workspace=tmp_path, device="cpu", seed=1, timeout=3)
    server = StatefulServer()
    client._opener = server
    first, second, third = context(0, 0), context(1, 1), context(2, 0)
    client.reset(first)
    client.reset(second)
    chunks = client.predict_batch([observation(), observation()], [second, first])
    assert [chunk["motion_token"].shape[0] for chunk in chunks] == [2, 1]
    client.end_episode(first)
    client.reset(third)
    chunks = client.predict_batch([observation(), observation()], [third, second])
    assert [chunk["motion_token"][0, 0] for chunk in chunks] == [1, 2]
    assert [chunk["motion_token"].shape[0] for chunk in chunks] == [3, 2]
    count = len(server.requests)
    with pytest.raises(ValueError, match="active reset episode"):
        client.predict_batch([observation()], [first])
    with pytest.raises(ValueError, match="active reset episode"):
        client.predict_batch([observation()], [replace(second, seed=999)])
    with pytest.raises(ValueError, match="distinct episodes"):
        client.predict_batch([observation(), observation()], [second, second])
    assert len(server.requests) == count
    client.close()
    assert not server.active
    client.close()


@pytest.mark.parametrize("raw", [
    b'{"protocol":"robocasa-g1-v1","fingerprint":"test-fingerprint","ok":true,"ok":false}',
    b'{"protocol":"another-protocol","fingerprint":"test-fingerprint","ok":true}',
    b'{"protocol":"robocasa-g1-v1","fingerprint":"test-fingerprint","ok":"true"}',
    b'{"protocol":"robocasa-g1-v1","fingerprint":"test-fingerprint","ok":true,"extra":true}',
    b'not JSON',
])
def test_http_requires_strict_protocol_and_acknowledgement(tmp_path, raw, service_metadata):
    client = HttpPolicy(endpoint="http://localhost", workspace=tmp_path, device="cpu", seed=0)
    client._opener = SimpleNamespace(open=lambda *a, **k: FakeResponse(raw=raw))
    with pytest.raises(ValueError):
        client.reset(context())
    assert client._contexts == {context().episode_id: context()}


def test_http_failed_reset_can_still_clean_up_remote_history(tmp_path, service_metadata):
    client = HttpPolicy(endpoint="http://localhost", workspace=tmp_path, device="cpu", seed=0)
    requests = []

    def open_request(request, **kwargs):
        requests.append(request.full_url)
        if request.full_url.endswith("/reset"):
            # The server may have initialized history before the connection broke.
            raise urllib.error.URLError("lost acknowledgement")
        return FakeResponse({"protocol": PROTOCOL, "ok": True})

    client._opener = SimpleNamespace(open=open_request)
    with pytest.raises(RuntimeError):
        client.reset(context())
    client.end_episode(context())
    assert not client._contexts
    assert requests == ["http://localhost/reset", "http://localhost/end_episode"]


def test_http_failure_does_not_expose_endpoint_or_authentication(tmp_path, monkeypatch, service_metadata):
    secret = "private-credential"
    monkeypatch.setenv("TEST_VLA_SECRET", secret)
    client = HttpPolicy(endpoint="https://localhost/private-model", workspace=tmp_path, device="cpu", seed=0,
                        auth_token_env="TEST_VLA_SECRET")

    def fail(*args, **kwargs):
        raise urllib.error.URLError(f"failed https://localhost/private-model: {secret}")

    client._opener = SimpleNamespace(open=fail)
    with pytest.raises(RuntimeError) as raised:
        client.reset(context())
    assert secret not in str(raised.value)
    assert "private-model" not in str(raised.value)
    assert "request failed" in str(raised.value)
    with pytest.raises(RuntimeError, match="request failed"):
        client.close()
    assert "Authorization" not in client._headers
