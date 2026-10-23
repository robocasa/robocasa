"""Model-independent policy interface for Unitree G1 + SONIC evaluation.

The evaluator supplies raw observations and expects physical SONIC actions.
Normalization, temporal history, model-specific image transforms, and conversion
from another action space belong to the policy adapter, not the simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import numpy as np


STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
ACTION_DIMS = {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7}
POLICY_ALIASES = {
    "groot": "robocasa_eval.adapters.groot:create_policy",
    "http": "robocasa_eval.adapters.http:create_policy",
}


@dataclass(frozen=True)
class EpisodeContext:
    """Stable episode identity, including across dynamically batched requests.

    A slot can host multiple episodes over time. Stateful policies must key their
    history by ``episode_id`` and discard it in ``end_episode``. The evaluator
    passes the same context from reset through the final prediction/end call.
    """

    episode_id: str
    task: str
    episode: int
    seed: int
    rank: int
    slot: int


class BasePolicy(ABC):
    """Implement ``predict_batch``; override lifecycle hooks when stateful.

    Observations contain ``video`` (camera-name -> HWC uint8 RGB), ``state``
    (the named vectors in STATE_DIMS), and a plain ``language`` instruction.
    Return one action mapping per input, in the same order. Each mapping contains
    exactly the three ACTION_DIMS keys with arrays of shape (T, D), T >= 1.
    A policy may return different T for different inputs, or a single step.

    predict_batch may receive a different batch composition on every invocation.
    Only reset/end_episode define episode boundaries; batch indices do not.
    """

    def reset(self, context: EpisodeContext) -> None:
        """Initialize per-episode state before the first observation."""

    @abstractmethod
    def predict_batch(
        self, observations: Sequence[Mapping[str, Any]], contexts: Sequence[EpisodeContext]
    ) -> list[dict[str, np.ndarray]]:
        """Return denormalized G1 SONIC action chunks, preserving input order."""
        raise NotImplementedError

    def end_episode(self, context: EpisodeContext) -> None:
        """Release per-episode state after success, timeout, or an error."""

    def close(self) -> None:
        """Release model or transport resources when a worker exits."""


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def validate_action_batch(
    actions: Sequence[Mapping[str, Any]],
    batch_size: int,
    execution_horizon: int,
    remaining_steps: Sequence[int] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Validate and copy action chunks, truncating only after full validation.

    Each sample's three arrays must share a positive T. Every returned value,
    including a tail that will not execute, must be finite. Executed length is
    min(T, execution_horizon, remaining_steps[i]) when remaining_steps is given.
    Output arrays are independent C-contiguous float32 copies. Unknown keys,
    flat actions, implicit broadcasting, NaNs, and empty chunks are rejected.
    """
    batch_size = _positive_int(batch_size, "batch_size", allow_zero=True)
    execution_horizon = _positive_int(execution_horizon, "execution_horizon")
    if not isinstance(actions, (list, tuple)) or len(actions) != batch_size:
        raise ValueError(f"Policy must return a list/tuple of {batch_size} action mappings.")
    limits = [execution_horizon] * batch_size
    if remaining_steps is not None:
        if isinstance(remaining_steps, (str, bytes)):
            raise ValueError("remaining_steps must contain one positive integer per sample.")
        try:
            steps = list(remaining_steps)
        except TypeError as error:
            raise ValueError("remaining_steps must be a sequence.") from error
        if len(steps) != batch_size:
            raise ValueError("remaining_steps length must match batch_size.")
        limits = [
            min(execution_horizon, _positive_int(value, f"remaining_steps[{i}]"))
            for i, value in enumerate(steps)
        ]
    result = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or set(action) != set(ACTION_DIMS):
            raise ValueError(
                f"Action {index} must contain exactly {', '.join(ACTION_DIMS)}."
            )
        arrays = {}
        horizon = None
        for key, dim in ACTION_DIMS.items():
            raw = np.asarray(action[key])
            if raw.dtype.kind not in "iuf":
                raise ValueError(f"Action {index}.{key} must contain real numeric values.")
            if raw.ndim != 2 or raw.shape[1] != dim or raw.shape[0] < 1:
                raise ValueError(f"Action {index}.{key} must have shape (T, {dim}), T >= 1.")
            if horizon is not None and raw.shape[0] != horizon:
                raise ValueError(f"Action {index}: all three action chunks must share T.")
            horizon = raw.shape[0]
            if not np.isfinite(raw).all():
                raise ValueError(f"Action {index}.{key} contains non-finite values.")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(raw, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"Action {index}.{key} exceeds finite float32 range.")
            arrays[key] = np.array(array[: limits[index]], dtype=np.float32, order="C", copy=True)
        result.append(arrays)
    return result


def resolve_factory(factory_spec: str):
    """Resolve a factory without creating a policy or allocating model weights.

    Built-in adapter modules import no GPU runtime at module scope. Custom
    factories should follow the same convention so preflight remains lightweight.
    """
    if not isinstance(factory_spec, str) or not factory_spec.strip():
        raise ValueError("policy.factory must be a built-in alias or module:function.")
    factory_spec = POLICY_ALIASES.get(factory_spec, factory_spec)
    if factory_spec.count(":") != 1:
        raise ValueError("policy.factory must use the form module:function.")
    module_name, attribute = factory_spec.split(":")
    if not module_name or not attribute or not attribute.isidentifier():
        raise ValueError("policy.factory must use the form module:function.")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError(f"Policy factory is not callable: {factory_spec}")
    return factory


def load_policy(config: Mapping[str, Any], *, device: str, seed: int) -> BasePolicy:
    """Lazily construct a configured policy; importing this module needs no GPU.

    config must define a workspace and a policy mapping. policy.factory is a
    built-in alias or ``importable.module:function``; policy.kwargs holds only
    adapter options. Every factory receives device, seed, and the resolved
    workspace explicitly. A duck-typed implementation may provide the four
    lifecycle methods without subclassing BasePolicy.
    """
    policy_config = config.get("policy")
    if not isinstance(policy_config, Mapping):
        raise ValueError("config.policy must be a mapping.")
    # id/artifacts belong to the run manifest/resume fingerprint, never model
    # constructor arguments. Keeping them here supports the normalized config.
    unknown = set(policy_config) - {"factory", "kwargs", "id", "artifacts"}
    if unknown:
        raise ValueError(f"Unknown policy configuration fields: {sorted(unknown)}")
    factory_spec = policy_config.get("factory")
    kwargs = policy_config.get("kwargs", {})
    if not isinstance(kwargs, Mapping) or any(not isinstance(key, str) for key in kwargs):
        raise ValueError("policy.kwargs must be a mapping with string keys.")
    reserved = {"device", "seed", "workspace"} & set(kwargs)
    if reserved:
        raise ValueError(f"policy.kwargs cannot override {', '.join(sorted(reserved))}.")
    workspace = config.get("workspace")
    if not isinstance(workspace, (str, Path)) or not str(workspace).strip():
        raise ValueError("An explicit config.workspace directory is required.")
    workspace = Path(workspace).expanduser().resolve()
    factory = resolve_factory(factory_spec)
    policy = factory(device=device, seed=seed, workspace=workspace, **dict(kwargs))
    for name in ("reset", "predict_batch", "end_episode", "close"):
        if not callable(getattr(policy, name, None)):
            close = getattr(policy, "close", None)
            if callable(close):
                close()
            raise TypeError(f"Policy factory result must implement {name}().")
    return policy
