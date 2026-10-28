"""Configuration for a VLA inference service, independent of simulation assets."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

from .configuration import HERE, normalize_policy, policy_identity, sha256_file
from .evaluation import digest


DEFAULTS = {
    "workspace": "./evaluation_runs", "host": "127.0.0.1", "port": 8000,
    "devices": ["cpu"], "seed": 20261020, "max_batch_size": 32,
    "batch_wait_ms": 5, "request_timeout_seconds": 120,
    "read_timeout_seconds": 30, "max_request_bytes": 64 * 1024**2,
    "max_active_episodes": 4096, "idle_session_seconds": 600,
    "auth_token_env": None, "disk_limit_bytes": 2_000_000_000_000,
    "minimum_free_bytes": 10 * 1024**3,
}


def build_server_config(options, *, verify_files=False):
    if not isinstance(options, dict) or "policy" not in options:
        raise ValueError("serve requires a configuration with an explicit policy")
    unknown = set(options) - (set(DEFAULTS) | {"policy"})
    if unknown:
        raise ValueError(f"Unknown model service options: {sorted(unknown)}")
    config = dict(copy.deepcopy(DEFAULTS), **copy.deepcopy(options))
    root = config["workspace"]
    if not isinstance(root, str) or not root.strip():
        raise ValueError("workspace must be a directory path")
    config["workspace"] = str(Path(root).expanduser().resolve())
    if not isinstance(config["host"], str) or not config["host"].strip():
        raise ValueError("host must be a hostname or bind address")
    for key, minimum in (("port", 1), ("seed", 0), ("max_batch_size", 1),
                         ("batch_wait_ms", 0), ("max_active_episodes", 1),
                         ("disk_limit_bytes", 1), ("minimum_free_bytes", 0),
                         ("max_request_bytes", 1)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["port"] > 65535:
        raise ValueError("port must be 1..65535")
    if config["max_batch_size"] > 4096 or config["max_request_bytes"] > 64 * 1024**2:
        raise ValueError("Model service batch or request limit is too large")
    for key in ("request_timeout_seconds", "read_timeout_seconds", "idle_session_seconds"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    devices = config["devices"]
    if (not isinstance(devices, list) or not devices or any(not isinstance(d, str) or
            re.fullmatch(r"(?:cpu|[a-z][a-z0-9_]*:[0-9]+)", d) is None for d in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("devices must contain distinct devices such as cpu or cuda:0")
    auth = config["auth_token_env"]
    if auth is not None and (not isinstance(auth, str) or not re.fullmatch(r"[A-Za-z_]\w*", auth)):
        raise ValueError("auth_token_env must name an environment variable")
    config["policy"] = normalize_policy(config["policy"])
    if config["policy"]["factory"] in {"http", "robocasa_eval.adapters.http:create_policy"}:
        raise ValueError("serve requires a local model adapter, not another HTTP client")
    config["policy_identity"] = policy_identity(config["policy"], verify_files, config["workspace"])
    config["code_sha256"] = {str(p.relative_to(HERE)): sha256_file(p) for p in sorted(HERE.rglob("*.py"))}
    config["identity"] = digest(config)
    return config


def service_metadata(config):
    from .adapters.http import PROTOCOL
    from .policy import ACTION_DIMS
    # Replica counts/devices may change after a single-GPU smoke. The protocol
    # gate pins the model, device family and batching, not deployment topology.
    identity = {key: config[key] for key in (
        "policy", "policy_identity", "code_sha256", "seed",
        "max_batch_size", "batch_wait_ms",
    )}
    identity["device_families"] = sorted({device.split(":", 1)[0] for device in config["devices"]})
    metadata = dict(protocol=PROTOCOL, fingerprint=digest(identity),
                    action_dims=dict(ACTION_DIMS), replicas=len(config["devices"]))
    if config["policy"].get("id") is not None:
        metadata["model_id"] = config["policy"]["id"]
    return metadata


def bind_remote_policy(config):
    """Pin a live inference revision before rollouts, without requiring a name."""
    policy = config["policy"]
    if policy["factory"] not in {"http", "robocasa_eval.adapters.http:create_policy"}:
        return None
    from .adapters.http import fetch_metadata
    kwargs = policy["kwargs"]
    if not kwargs.get("endpoint"):
        raise ValueError("HTTP evaluation requires policy.kwargs.endpoint")
    explicit_id = policy.get("id")
    expected_model = kwargs.get("model_id", explicit_id)
    if explicit_id is not None and expected_model != explicit_id:
        raise ValueError("HTTP model_id must match policy.id")
    metadata = fetch_metadata(kwargs["endpoint"], timeout=kwargs.get("timeout", 120),
                              auth_token_env=kwargs.get("auth_token_env"))
    if expected_model is not None and metadata.get("model_id") != expected_model:
        raise ValueError("Model service identity does not match policy.id")
    if kwargs.get("fingerprint") not in (None, metadata["fingerprint"]):
        raise ValueError("Inference service fingerprint changed; use the intended service")
    config["remote_policy_identity"] = {key: metadata[key] for key in
        ("protocol", "fingerprint", "action_dims")}
    if expected_model is not None:
        config["remote_policy_identity"]["model_id"] = expected_model
        kwargs["model_id"] = expected_model
    kwargs["fingerprint"] = metadata["fingerprint"]
    return metadata
