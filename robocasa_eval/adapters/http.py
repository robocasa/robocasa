"""HTTP transport for the RoboCasa G1 policy protocol (version 1).

This is a small project-specific JSON protocol, not an OpenPI or GR00T server
protocol. Implement a server shim to use an existing model service.

The model opens one HTTP port; evaluation clients connect to that port. GET
``/metadata`` identifies the loaded model, and ``/health`` reports readiness.
POST requests use Content-Type application/json. Every request and response
contains ``"protocol": "robocasa-g1-v1"``. Endpoint suffixes are:

* /reset: request {protocol, context}; response {protocol, fingerprint, ok: true}.
* /predict: request {protocol, observations: [...], contexts: [...]}; response
  {protocol, fingerprint, actions: [...]}, with actions in the same order as contexts.
* /end_episode: request {protocol, context}; response {protocol, fingerprint, ok: true}.

Clients verified against /metadata send X-Robocasa-Model-Fingerprint on each
request and verify the same fingerprint in each response. Legacy clients that
omit an expected model identity can still communicate with unextended v1 peers.

A context is EpisodeContext serialized as an object. Servers must isolate
history by episode_id, not batch position or GPU slot. State survives changes in
batch composition until /end_episode. Each rank owns one client; a shared server
must support multiple concurrent clients and globally unique episode IDs.

Arrays are objects with exactly four keys: {"__ndarray__": true, "dtype": "<f4",
"shape": [T, 64], "data": "<base64 C-order bytes>"}. Images use dtype "|u1"
and shape [H, W, 3]. Supported dtypes are bool, integers, and floats up to 64 bits.
JSON numeric lists also work for action arrays. Non-finite numbers are forbidden.
Import encode_value/decode_value to implement a compatible Python server.

The server returns denormalized, absolute G1 SONIC motion tokens and hand
positions in the same thumb-first order as the local GROOT adapter. An EEF,
Panda, or differently embodied policy requires an explicit action adapter.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from robocasa_eval.policy import ACTION_DIMS, BasePolicy, EpisodeContext, validate_action_batch


PROTOCOL = "robocasa-g1-v1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_ARRAY_KEYS = {"__ndarray__", "dtype", "shape", "data"}
FINGERPRINT_HEADER = "X-Robocasa-Model-Fingerprint"


def encode_value(value):
    """Encode nested observations/actions without lossy image conversion."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif" or value.dtype.itemsize > 8:
            raise ValueError("HTTP arrays must use real numeric or boolean dtypes <= 64 bits.")
        if not np.isfinite(value).all():
            raise ValueError("HTTP arrays must contain finite values.")
        if value.nbytes > MAX_MESSAGE_BYTES:
            raise ValueError("HTTP array exceeds the message size limit.")
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("HTTP object keys must be strings.")
        if "__ndarray__" in value:
            raise ValueError("__ndarray__ is reserved for encoded array objects.")
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("HTTP values must be finite.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported HTTP value type: {type(value).__name__}")


def decode_value(value):
    """Decode and validate the protocol's base64 arrays into independent copies."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            if set(value) != _ARRAY_KEYS or value["__ndarray__"] is not True:
                raise ValueError("Malformed HTTP array object.")
            if not isinstance(value["dtype"], str) or not isinstance(value["data"], str):
                raise ValueError("HTTP array dtype and data must be strings.")
            try:
                dtype = np.dtype(value["dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid HTTP array dtype.") from error
            if dtype.kind not in "buif" or dtype.itemsize > 8:
                raise ValueError("HTTP array dtype must be real numeric or boolean <= 64 bits.")
            shape = value["shape"]
            if not isinstance(shape, list) or len(shape) > 8 or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0
                or n > MAX_MESSAGE_BYTES for n in shape
            ):
                raise ValueError("Invalid HTTP array shape.")
            expected_bytes = math.prod(shape) * dtype.itemsize
            if expected_bytes > MAX_MESSAGE_BYTES or len(value["data"]) > MAX_MESSAGE_BYTES * 2:
                raise ValueError("HTTP array exceeds the message size limit.")
            try:
                payload = base64.b64decode(value["data"], validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid HTTP array base64 data.") from error
            if len(payload) != expected_bytes:
                raise ValueError("HTTP array byte count does not match dtype and shape.")
            array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
            if not np.isfinite(array).all():
                raise ValueError("HTTP array contains non-finite values.")
            return array
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("HTTP values must be finite.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Invalid HTTP JSON value.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object fields are not allowed.")
        result[key] = value
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials must never follow an unexpected cross-host redirect.
        raise urllib.error.HTTPError(req.full_url, code, "Redirect rejected", headers, fp)


def _connection(endpoint, timeout, auth_token_env):
    if not isinstance(endpoint, str):
        raise ValueError("HTTP policy endpoint must be a URL string.")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.netloc
        or parsed.username or parsed.password or parsed.query or parsed.fragment
    ):
        raise ValueError("HTTP endpoint must be http(s), without userinfo, query, or fragment.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HTTP timeout must be a positive finite number.")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth_token_env is not None:
        if not isinstance(auth_token_env, str) or not auth_token_env:
            raise ValueError("auth_token_env must name an environment variable.")
        token = os.environ.get(auth_token_env)
        if not token or any(character in token for character in "\r\n"):
            raise ValueError("HTTP authentication environment variable is missing or invalid.")
        headers["Authorization"] = "Bearer " + token
    return endpoint.rstrip("/"), float(timeout), headers


def _read_response(opener, request, timeout, suffix):
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP policy {suffix} returned status {response.status}.")
            raw = response.read(MAX_MESSAGE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP policy {suffix} failed with status {error.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError(f"HTTP policy {suffix} request failed.") from None
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("HTTP response exceeds the message size limit.")
    try:
        result = decode_value(json.loads(raw, object_pairs_hook=_unique_object))
    except (UnicodeError, ValueError, TypeError):
        raise ValueError(f"HTTP policy {suffix} returned malformed JSON/array data.") from None
    if not isinstance(result, dict) or result.get("protocol") != PROTOCOL:
        raise ValueError(f"HTTP policy {suffix} returned an incompatible protocol.")
    return result


def validate_metadata(metadata):
    """Validate the model identity published on the inference port."""
    if not isinstance(metadata, dict) or metadata.get("protocol") != PROTOCOL:
        raise ValueError("HTTP model metadata has an incompatible protocol.")
    for key in ("model_id", "fingerprint"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
            raise ValueError(f"HTTP model metadata requires a nonempty {key}.")
    dims = metadata.get("action_dims")
    if not isinstance(dims, dict) or dims != ACTION_DIMS or any(type(v) is not int for v in dims.values()):
        raise ValueError("HTTP model metadata has incompatible G1 action dimensions.")
    replicas = metadata.get("replicas")
    if type(replicas) is not int or replicas < 1:
        raise ValueError("HTTP model metadata replicas must be a positive integer.")
    return metadata


def fetch_metadata(endpoint, *, timeout=120, auth_token_env=None):
    """Read the serving model identity without resetting or loading a policy."""
    endpoint, timeout, headers = _connection(endpoint, timeout, auth_token_env)
    request = urllib.request.Request(endpoint + "/metadata", headers=headers, method="GET")
    result = _read_response(urllib.request.build_opener(_NoRedirect()), request, timeout, "/metadata")
    return validate_metadata(result)


class HttpPolicy(BasePolicy):
    """Synchronous batched HTTP client with explicit per-episode lifecycle."""

    def __init__(self, *, endpoint, workspace, device, seed, timeout=120, auth_token_env=None,
                 model_id=None, fingerprint=None):
        self._endpoint, self.timeout, self._headers = _connection(endpoint, timeout, auth_token_env)
        self.workspace = Path(workspace).expanduser().resolve()
        self.device = device
        self.seed = seed
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._contexts = {}
        self._closed = False
        self.metadata = None
        self._fingerprint = None
        if model_id is not None or fingerprint is not None:
            for key, value in (("model_id", model_id), ("fingerprint", fingerprint)):
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"HTTP expected {key} must be a nonempty string.")
            metadata = fetch_metadata(endpoint, timeout=timeout, auth_token_env=auth_token_env)
            if model_id is not None and metadata["model_id"] != model_id:
                raise ValueError("HTTP serving model_id does not match the configured model.")
            if fingerprint is not None and metadata["fingerprint"] != fingerprint:
                raise ValueError("HTTP serving fingerprint does not match the verified model.")
            self.metadata = metadata
            self._fingerprint = metadata["fingerprint"]
            self._headers[FINGERPRINT_HEADER] = self._fingerprint

    def _request(self, suffix, payload):
        if self._closed:
            raise RuntimeError("HTTP policy is closed.")
        body = json.dumps(
            {"protocol": PROTOCOL, **encode_value(payload)},
            allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_MESSAGE_BYTES:
            raise ValueError("HTTP request exceeds the message size limit.")
        request = urllib.request.Request(
            self._endpoint + suffix, data=body, headers=self._headers, method="POST"
        )
        result = _read_response(self._opener, request, self.timeout, suffix)
        if self._fingerprint is not None and result.get("fingerprint") != self._fingerprint:
            raise ValueError("HTTP response fingerprint differs from the verified model.")
        # Fingerprints extend the v1 response while retaining compatibility with
        # older peers when a caller explicitly constructs an unpinned client.
        if "fingerprint" in result:
            value = result.pop("fingerprint")
            if not isinstance(value, str) or not value:
                raise ValueError("HTTP response fingerprint must be a nonempty string.")
        return result

    def _ack(self, suffix, context):
        result = self._request(suffix, {"context": asdict(context)})
        if set(result) != {"protocol", "ok"} or result["ok"] is not True:
            raise ValueError(f"HTTP policy {suffix} must acknowledge with ok=true.")

    def reset(self, context: EpisodeContext) -> None:
        if not isinstance(context, EpisodeContext):
            raise TypeError("reset requires an EpisodeContext.")
        if context.episode_id in self._contexts:
            raise ValueError("Episode is already active; end it before resetting.")
        # Register before sending: a lost acknowledgement may still mean the
        # server initialized history. The evaluator can then call end_episode
        # even when reset raised, and close can clean up a standalone client.
        self._contexts[context.episode_id] = context
        self._ack("/reset", context)

    def _check_context(self, context):
        if not isinstance(context, EpisodeContext) or self._contexts.get(context.episode_id) != context:
            raise ValueError("HTTP policy context must match an active reset episode.")

    def predict_batch(self, observations, contexts):
        if len(observations) != len(contexts):
            raise ValueError("Observations and contexts must have the same batch length.")
        if self._closed:
            raise RuntimeError("HTTP policy is closed.")
        if not observations:
            return []
        for context in contexts:
            self._check_context(context)
        if len({context.episode_id for context in contexts}) != len(contexts):
            raise ValueError("A prediction batch must contain distinct episodes.")
        result = self._request("/predict", {
            "observations": observations,
            "contexts": [asdict(context) for context in contexts],
        })
        if set(result) != {"protocol", "actions"}:
            raise ValueError("HTTP predict response must contain only protocol and actions.")
        # Validate here for direct users too; the evaluator applies its execution
        # horizon separately. The transport imposes no model-specific chunk size.
        return validate_action_batch(result["actions"], len(observations), 2**63 - 1)

    def end_episode(self, context: EpisodeContext) -> None:
        self._check_context(context)
        self._ack("/end_episode", context)
        del self._contexts[context.episode_id]

    def close(self) -> None:
        if self._closed:
            return
        first_error = None
        try:
            for context in list(self._contexts.values()):
                try:
                    self.end_episode(context)
                except Exception as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._contexts.clear()
            self._headers.pop("Authorization", None)
            self._closed = True
        if first_error is not None:
            raise first_error


def create_policy(*, device, seed, workspace, endpoint, timeout=120, auth_token_env=None,
                  model_id=None, fingerprint=None):
    return HttpPolicy(
        endpoint=endpoint, workspace=workspace, device=device, seed=seed,
        timeout=timeout, auth_token_env=auth_token_env, model_id=model_id, fingerprint=fingerprint,
    )
