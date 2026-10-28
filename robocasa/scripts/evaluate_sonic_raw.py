#!/usr/bin/env python3
"""Evaluate raw SONIC episodes by replaying recorded MuJoCo states.

The evaluator is task-independent: it restores each episode's recorded model
and metadata, then calls that environment's own ``_check_success()`` method.
It never steps recorded actions. This makes the result a task-state replay
check, rather than a test of controller / dynamics determinism.

Evaluation is read-only by default. ``--delete-failed`` removes episodes from
the active dataset by atomically moving fully evaluated failures and
deterministically invalid episode data into a sibling quarantine directory.
Unexpected loading or replay errors are never moved. The JSON report and
cleanup plan are fsynced before any move occurs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shlex
import signal
import subprocess
import sys
import traceback
from typing import Any

import h5py
import numpy as np


EPISODE_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
TIMESTAMPED_TASK_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_sonic$"
)
STATE_FILE_PATTERN = re.compile(r"^state_(?P<seconds>\d+)_(?P<fraction>\d+)\.npz$")
REPORT_SCHEMA_VERSION = 1


class EvaluationError(RuntimeError):
    """Raised for invalid input or an episode that cannot be evaluated."""


@dataclasses.dataclass(frozen=True)
class EpisodeCandidate:
    """One episode directory selected for evaluation."""

    episode_dir: Path
    hdf5_path: Path | None
    task_hint: str | None
    discovery_error: str | None = None


@dataclasses.dataclass
class EpisodeData:
    """Metadata and state trajectory loaded from one per-episode HDF5."""

    candidate: EpisodeCandidate
    task: str
    env_meta: dict[str, Any]
    ep_meta: dict[str, Any]
    model_xml: str
    states: np.ndarray
    state_source: str
    state_files: list[str]
    hdf5_state_count: int
    control_freq: float
    post_action_freq: int
    sonic_runtime_text: str | bytes | None
    sonic_gains_text: str | bytes | None

    @property
    def instruction(self) -> str:
        value = self.ep_meta.get("lang", "")
        return str(value) if value is not None else ""


def _decode_text(value: Any, *, label: str, path: Path) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{path}: {label} must be a non-empty string")
    return value


def _decode_json(value: Any, *, label: str, path: Path) -> dict[str, Any]:
    text = _decode_text(value, label=label, path=path)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: invalid {label} JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError(f"{path}: {label} must decode to an object")
    return decoded


def _task_hint_for_episode(episode_dir: Path) -> str | None:
    for parent in episode_dir.parents:
        if parent.name.endswith("_sonic"):
            timestamped = TIMESTAMPED_TASK_PATTERN.fullmatch(parent.name)
            if timestamped is not None:
                return timestamped.group("task")
            task = parent.name[: -len("_sonic")]
            if task:
                return task
    return None


def _walk_episode_directories(root: Path) -> list[EpisodeCandidate]:
    candidates: list[EpisodeCandidate] = []
    for current_root, directory_names, _ in os.walk(root, followlinks=False):
        current = Path(current_root)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            if EPISODE_PATTERN.fullmatch(name):
                if path.is_symlink():
                    candidates.append(
                        EpisodeCandidate(
                            episode_dir=path,
                            hdf5_path=None,
                            task_hint=_task_hint_for_episode(path),
                            discovery_error="episode directory is a symlink",
                        )
                    )
                elif path.is_dir():
                    hdf5_path = path / "ep_demo.hdf5"
                    candidates.append(
                        EpisodeCandidate(
                            episode_dir=path,
                            hdf5_path=hdf5_path if hdf5_path.is_file() else None,
                            task_hint=_task_hint_for_episode(path),
                        )
                    )
                continue
            if path.is_symlink():
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
    return candidates


def discover_episodes(
    dataset: Path | str,
    *,
    tasks: Sequence[str] = (),
    episode_names: Sequence[str] = (),
) -> list[EpisodeCandidate]:
    """Discover direct or nested ``ep_*`` directories in stable order."""

    input_path = Path(dataset).expanduser()
    if input_path.is_symlink():
        raise EvaluationError(f"dataset path must not be a symlink: {input_path}")
    try:
        resolved = input_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvaluationError(f"dataset does not exist: {input_path}") from exc

    if resolved.is_file():
        if resolved.suffix not in {".hdf5", ".h5"}:
            raise EvaluationError(f"dataset file is not HDF5: {resolved}")
        candidate = EpisodeCandidate(
            episode_dir=resolved.parent,
            hdf5_path=resolved,
            task_hint=_task_hint_for_episode(resolved.parent),
        )
        candidates = [candidate]
    elif EPISODE_PATTERN.fullmatch(resolved.name):
        hdf5_path = resolved / "ep_demo.hdf5"
        candidates = [
            EpisodeCandidate(
                episode_dir=resolved,
                hdf5_path=hdf5_path if hdf5_path.is_file() else None,
                task_hint=_task_hint_for_episode(resolved),
            )
        ]
    elif resolved.is_dir():
        candidates = _walk_episode_directories(resolved)
    else:
        raise EvaluationError(f"unsupported dataset path: {resolved}")

    requested_tasks = set(tasks)
    requested_episodes = set(episode_names)
    if requested_tasks:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.task_hint in requested_tasks
        ]
    if requested_episodes:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.episode_dir.name in requested_episodes
        ]
    candidates.sort(
        key=lambda candidate: (
            candidate.task_hint or "",
            candidate.episode_dir.name,
            str(candidate.episode_dir),
        )
    )
    if not candidates:
        raise EvaluationError(
            f"no episode directories matched dataset={resolved}, "
            f"tasks={sorted(requested_tasks)}, episodes={sorted(requested_episodes)}"
        )
    return candidates


def _state_file_sort_key(path: Path) -> tuple[int, str, str]:
    match = STATE_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        return (sys.maxsize, path.name, path.name)
    fraction = match.group("fraction")
    # Right padding preserves decimal ordering even if the fractional component
    # was emitted with a different number of digits.
    normalized_fraction = fraction[:18].ljust(18, "0")
    return (int(match.group("seconds")), normalized_fraction, path.name)


def _load_raw_states(
    episode_dir: Path,
    *,
    expected_count: int,
    expected_width: int,
) -> tuple[np.ndarray | None, list[str]]:
    paths = sorted(episode_dir.glob("state_*.npz"), key=_state_file_sort_key)
    if not paths:
        return None, []

    chunks: list[np.ndarray] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(f"raw state file must be a regular file: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                states = np.asarray(archive["states"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            raise EvaluationError(f"cannot read raw states from {path}: {exc}") from exc
        if states.ndim != 2 or states.shape[0] == 0:
            raise EvaluationError(f"{path}: states must be a non-empty 2D array")
        if states.shape[1] != expected_width:
            raise EvaluationError(
                f"{path}: state width {states.shape[1]} != HDF5 width {expected_width}"
            )
        chunks.append(states)

    states = np.concatenate(chunks, axis=0)
    if states.shape[0] != expected_count + 1:
        raise EvaluationError(
            f"{episode_dir}: raw state count {states.shape[0]} must equal "
            f"HDF5 state count + 1 ({expected_count + 1})"
        )
    return states, [str(path) for path in paths]


def load_episode(candidate: EpisodeCandidate) -> EpisodeData:
    """Load and validate an episode, preferring the true raw terminal state."""

    if candidate.discovery_error is not None:
        raise EvaluationError(
            f"{candidate.episode_dir}: {candidate.discovery_error}"
        )
    if candidate.hdf5_path is None:
        raise EvaluationError(
            f"{candidate.episode_dir}: missing ep_demo.hdf5"
        )
    hdf5_path = candidate.hdf5_path
    if hdf5_path.is_symlink() or not hdf5_path.is_file():
        raise EvaluationError(f"HDF5 must be a regular file: {hdf5_path}")

    try:
        hdf5_file = h5py.File(hdf5_path, "r")
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot open {hdf5_path}: {exc}") from exc

    try:
        with hdf5_file:
            if "data" not in hdf5_file:
                raise EvaluationError(f"{hdf5_path}: missing data group")
            data_group = hdf5_file["data"]
            demo_names = list(data_group.keys())
            if len(demo_names) != 1:
                raise EvaluationError(
                    f"{hdf5_path}: expected one demo, found {len(demo_names)}"
                )
            demo = data_group[demo_names[0]]
            if "states" not in demo:
                raise EvaluationError(f"{hdf5_path}: demo has no states dataset")
            hdf5_states = demo["states"]
            if hdf5_states.ndim != 2 or hdf5_states.shape[0] == 0:
                raise EvaluationError(
                    f"{hdf5_path}: states must be a non-empty 2D dataset"
                )
            hdf5_state_count = int(hdf5_states.shape[0])
            state_width = int(hdf5_states.shape[1])

            env_meta = _decode_json(
                data_group.attrs.get("env_args"),
                label="env_args",
                path=hdf5_path,
            )
            task = str(env_meta.get("env_name") or data_group.attrs.get("env") or "")
            if not task:
                raise EvaluationError(f"{hdf5_path}: environment task name is missing")
            if candidate.task_hint is not None and candidate.task_hint != task:
                raise EvaluationError(
                    f"{hdf5_path}: directory task {candidate.task_hint!r} "
                    f"does not match HDF5 task {task!r}"
                )
            ep_meta = _decode_json(
                demo.attrs.get("ep_meta"),
                label="ep_meta",
                path=hdf5_path,
            )
            model_xml = _decode_text(
                demo.attrs.get("model_file"),
                label="model_file",
                path=hdf5_path,
            )
            runtime_text = data_group.attrs.get("sonic_runtime")
            runtime = (
                _decode_json(
                    runtime_text,
                    label="sonic_runtime",
                    path=hdf5_path,
                )
                if runtime_text
                else {}
            )
            gains_text = data_group.attrs.get("sonic_gains")

            env_kwargs = env_meta.get("env_kwargs")
            if not isinstance(env_kwargs, dict):
                raise EvaluationError(f"{hdf5_path}: env_args.env_kwargs is invalid")
            control_freq = float(
                runtime.get("control_freq", env_kwargs.get("control_freq", 20))
            )
            if not np.isfinite(control_freq) or control_freq <= 0:
                raise EvaluationError(f"{hdf5_path}: invalid control frequency")
            post_action_freq = int(
                runtime.get("post_action_freq", max(1, round(control_freq / 20.0)))
            )
            if post_action_freq <= 0:
                raise EvaluationError(f"{hdf5_path}: invalid post_action_freq")

            raw_states, state_files = _load_raw_states(
                candidate.episode_dir,
                expected_count=hdf5_state_count,
                expected_width=state_width,
            )
            if raw_states is None:
                states = np.asarray(hdf5_states[()], dtype=np.float64)
                state_source = "ep_demo.hdf5/aligned_states"
            else:
                states = raw_states
                state_source = "raw_npz/true_terminal_state"
    except KeyError as exc:
        raise EvaluationError(f"{hdf5_path}: missing HDF5 key {exc}") from exc

    if not np.all(np.isfinite(states)):
        raise EvaluationError(f"{candidate.episode_dir}: states contain non-finite values")
    times = states[:, 0]
    if np.any(np.diff(times) < -1e-9):
        raise EvaluationError(f"{candidate.episode_dir}: state times are not monotonic")

    local_ep_meta = candidate.episode_dir / "ep_meta.json"
    local_model = candidate.episode_dir / "model.xml"
    if local_ep_meta.is_file():
        try:
            local_meta_value = json.loads(local_ep_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"cannot validate {local_ep_meta}: {exc}") from exc
        if local_meta_value != ep_meta:
            raise EvaluationError(f"{local_ep_meta}: differs from HDF5 ep_meta")
    if local_model.is_file() and local_model.read_text(encoding="utf-8") != model_xml:
        raise EvaluationError(f"{local_model}: differs from HDF5 model_file")

    return EpisodeData(
        candidate=candidate,
        task=task,
        env_meta=env_meta,
        ep_meta=ep_meta,
        model_xml=model_xml,
        states=states,
        state_source=state_source,
        state_files=state_files,
        hdf5_state_count=hdf5_state_count,
        control_freq=control_freq,
        post_action_freq=post_action_freq,
        sonic_runtime_text=runtime_text,
        sonic_gains_text=gains_text,
    )


def normalize_success(value: Any) -> bool:
    """Normalize RoboCasa bool and mapping-style success results."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Mapping):
        for key in ("task", "success"):
            if key in value:
                nested = value[key]
                if isinstance(nested, (bool, np.bool_)):
                    return bool(nested)
                if isinstance(nested, (int, float, np.integer, np.floating)):
                    return bool(nested)
                raise EvaluationError(
                    f"success mapping key {key!r} is not boolean-like: {nested!r}"
                )
        raise EvaluationError(
            f"success mapping has neither 'task' nor 'success': {sorted(value)}"
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    raise EvaluationError(f"unsupported success result: {type(value).__name__}")


def _set_sim_state(env: Any, state: np.ndarray, *, control_freq: float) -> int:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    timestep = int(round(float(state[0]) * control_freq))
    env.timestep = timestep
    env.cur_time = float(state[0])
    if hasattr(env, "update_sites"):
        env.update_sites()
    return timestep


def replay_success_checks(
    env: Any,
    episode: EpisodeData,
    *,
    success_mode: str,
) -> dict[str, Any]:
    """Replay checkpoint states and run the task's own success predicate.

    The first recorded state is the pre-action capture point, so checks begin at
    index 1. At normal checkpoints, the order matches live RoboCasa execution:
    ``_check_success()`` first, followed by ``update_state()``. A non-checkpoint
    terminal state is additionally probed so a save between throttled checks is
    still assessed; this is called out explicitly in the report.
    """

    if success_mode not in {"any", "final"}:
        raise EvaluationError(f"unsupported success mode: {success_mode}")
    if not hasattr(env, "_check_success"):
        raise EvaluationError(
            f"environment {type(env).__name__} has no _check_success()"
        )

    states = episode.states
    checkpoint_indices: list[int] = []
    for index in range(1, len(states)):
        timestep = int(round(float(states[index, 0]) * episode.control_freq))
        if timestep % episode.post_action_freq == 0:
            checkpoint_indices.append(index)
    terminal_probe = False
    terminal_index = len(states) - 1
    if terminal_index not in checkpoint_indices:
        checkpoint_indices.append(terminal_index)
        terminal_probe = True
    if not checkpoint_indices:
        checkpoint_indices = [0]
        terminal_probe = True

    checks: list[dict[str, Any]] = []
    for index in checkpoint_indices:
        timestep = _set_sim_state(
            env,
            states[index],
            control_freq=episode.control_freq,
        )
        success = normalize_success(env._check_success())
        checks.append(
            {
                "state_index": index,
                "sim_time": float(states[index, 0]),
                "timestep": timestep,
                "success": success,
                "terminal_probe": bool(
                    terminal_probe and index == terminal_index
                ),
            }
        )
        # Kitchen._post_action calls reward/_check_success before update_state.
        if hasattr(env, "update_state"):
            env.update_state()

    any_success = any(check["success"] for check in checks)
    final_success = bool(checks[-1]["success"])
    selected_success = any_success if success_mode == "any" else final_success
    successful_checks = [check for check in checks if check["success"]]
    return {
        "success": selected_success,
        "success_mode": success_mode,
        "any_success": any_success,
        "final_success": final_success,
        "checkpoint_count": len(checks),
        "first_success": successful_checks[0] if successful_checks else None,
        "last_success": successful_checks[-1] if successful_checks else None,
        "terminal_probe_added": terminal_probe,
        "first_state_time": float(states[0, 0]),
        "terminal_state_time": float(states[-1, 0]),
        "terminal_state_index": terminal_index,
    }


def _restore_episode(env: Any, episode: EpisodeData) -> None:
    """Reload exact episode XML/meta without playback reset_to's extra update."""

    import robosuite

    ep_meta = deepcopy(episode.ep_meta)
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    else:
        raise EvaluationError(
            f"environment {type(env).__name__} cannot restore episode metadata"
        )
    env.reset()
    robosuite_minor = int(robosuite.__version__.split(".")[1])
    if robosuite_minor <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(episode.model_xml)
    else:
        xml = env.edit_model_xml(episode.model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    # Restore runtime values that are not part of robomimic env_args. There is
    # deliberately no action step or controller-divergence check here.
    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
        _apply_sonic_runtime,
    )

    _apply_sonic_runtime(
        env,
        {"states": episode.states[0]},
        episode.sonic_gains_text,
        episode.sonic_runtime_text,
        integration_states=None,
        require_action_replay_metadata=False,
    )
    env.post_action_freq = episode.post_action_freq
    _set_sim_state(env, episode.states[0], control_freq=episode.control_freq)


class EnvironmentCache:
    """Reuse an environment while the complete construction metadata matches."""

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self.signature: str | None = None
        self.env: Any | None = None

    def _signature_for(self, episode: EpisodeData) -> str:
        return json.dumps(
            {
                "env_meta": episode.env_meta,
                "control_freq": episode.control_freq,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def get(self, episode: EpisodeData) -> Any:
        signature = self._signature_for(episode)
        if self.env is not None and signature == self.signature:
            return self.env
        self.close()

        # Importing robocasa registers its tasks with robosuite.
        __import__("robocasa")
        import robosuite
        import robosuite.macros as macros

        runtime = (
            _decode_json(
                episode.sonic_runtime_text,
                label="sonic_runtime",
                path=episode.candidate.hdf5_path or episode.candidate.episode_dir,
            )
            if episode.sonic_runtime_text
            else {}
        )
        if "sim_dt" in runtime:
            macros.SIMULATION_TIMESTEP = float(runtime["sim_dt"])

        env_kwargs = deepcopy(episode.env_meta["env_kwargs"])
        env_kwargs.update(
            env_name=episode.task,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            renderer="mjviewer",
            ignore_done=True,
            seed=self.seed,
        )
        self.env = robosuite.make(**env_kwargs)
        self.signature = signature
        return self.env

    def discard(self) -> None:
        self.close()

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
                self.signature = None


def evaluate_candidate(
    candidate: EpisodeCandidate,
    env_cache: EnvironmentCache,
    *,
    success_mode: str,
) -> dict[str, Any]:
    """Evaluate one candidate and return a JSON-serializable result."""

    base = {
        "episode": candidate.episode_dir.name,
        "episode_dir": str(candidate.episode_dir),
        "hdf5": str(candidate.hdf5_path) if candidate.hdf5_path else None,
        "task": candidate.task_hint,
    }
    if candidate.discovery_error is not None:
        return {
            **base,
            "status": "error",
            "error": candidate.discovery_error,
        }
    if candidate.hdf5_path is None:
        return {
            **base,
            "status": "missing_hdf5",
            "error": "episode has no ep_demo.hdf5",
        }

    try:
        episode = load_episode(candidate)
    except EvaluationError as exc:
        env_cache.discard()
        return {
            **base,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        env_cache.discard()
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    try:
        env = env_cache.get(episode)
        _restore_episode(env, episode)
        replay = replay_success_checks(env, episode, success_mode=success_mode)
        layout_id = episode.ep_meta.get("layout_id")
        style_id = episode.ep_meta.get("style_id")
        return {
            **base,
            "task": episode.task,
            "status": "passed" if replay["success"] else "failed",
            "instruction": episode.instruction,
            "layout_id": layout_id,
            "style_id": style_id,
            "state_source": episode.state_source,
            "state_files": episode.state_files,
            "state_count": int(episode.states.shape[0]),
            "hdf5_state_count": episode.hdf5_state_count,
            "control_freq": episode.control_freq,
            "post_action_freq": episode.post_action_freq,
            **replay,
        }
    except Exception as exc:
        env_cache.discard()
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_candidate_isolated(
    candidate: EpisodeCandidate,
    *,
    success_mode: str,
    seed: int,
    timeout_seconds: float,
    report_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Evaluate one episode in a fresh interpreter and validate its report."""

    base = {
        "episode": candidate.episode_dir.name,
        "episode_dir": str(candidate.episode_dir),
        "hdf5": str(candidate.hdf5_path) if candidate.hdf5_path else None,
        "task": candidate.task_hint,
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        str(candidate.episode_dir),
        "--episode",
        candidate.episode_dir.name,
        "--limit",
        "1",
        "--success-mode",
        success_mode,
        "--seed",
        str(seed),
        "--report",
        str(report_path),
    ]
    if candidate.task_hint is not None:
        command.extend(("--task", candidate.task_hint))
    started_at = dt.datetime.now().astimezone().isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    timed_out = False
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            stream.write("$ " + shlex.join(command) + "\n")
            stream.flush()
            child_env = os.environ.copy()
            child_env["PYTHONHASHSEED"] = str(seed)
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=child_env,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    except OSError as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return {
            **base,
            "status": "error",
            "error": f"IsolatedProcessError: cannot run evaluator: {exc}",
            "isolation": {"log": str(log_path), "report": str(report_path)},
        }

    completed_at = dt.datetime.now().astimezone().isoformat()
    returncode = int(returncode)
    isolation = {
        "started_at": started_at,
        "completed_at": completed_at,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "signal": -returncode if returncode < 0 else None,
        "log": str(log_path),
        "report": str(report_path),
    }
    isolation["log_sha256"] = _file_sha256(log_path)
    if report_path.is_file() and not report_path.is_symlink():
        isolation["report_sha256"] = _file_sha256(report_path)
    if timed_out:
        return {
            **base,
            "status": "error",
            "error": (
                "IsolatedProcessError: evaluator timed out after "
                f"{timeout_seconds:g} seconds"
            ),
            "isolation": isolation,
        }
    try:
        child = json.loads(report_path.read_text(encoding="utf-8"))
        child_results = child.get("results")
        child_summary = child.get("summary")
        if not isinstance(child_results, list) or len(child_results) != 1:
            raise EvaluationError("isolated report must contain exactly one result")
        if child.get("evaluated_count") != 1:
            raise EvaluationError("isolated report evaluated_count must be one")
        if child.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise EvaluationError("isolated report schema differs from parent")
        if child.get("success_mode") != success_mode or child.get("seed") != seed:
            raise EvaluationError("isolated report configuration differs from parent")
        if Path(child.get("dataset_input", "")).resolve() != candidate.episode_dir.resolve():
            raise EvaluationError("isolated report dataset_input differs from candidate")
        if Path(child.get("dataset_root", "")).resolve() != _dataset_root_for(
            candidate.episode_dir
        ):
            raise EvaluationError("isolated report dataset_root differs from candidate")
        selection = child.get("selection")
        expected_tasks = [candidate.task_hint] if candidate.task_hint else []
        if not isinstance(selection, Mapping) or selection.get("tasks") != expected_tasks:
            raise EvaluationError("isolated report task selection differs from candidate")
        if selection.get("episodes") != [candidate.episode_dir.name]:
            raise EvaluationError("isolated report episode selection differs from candidate")
        if selection.get("limit") != 1 or selection.get("candidate_count") != 1:
            raise EvaluationError("isolated report selection count is invalid")
        cleanup = child.get("cleanup")
        if (
            not isinstance(cleanup, Mapping)
            or cleanup.get("requested") is not False
            or cleanup.get("status") != "not_started"
        ):
            raise EvaluationError("isolated report cleanup is not report-only")
        if not isinstance(child_summary, Mapping):
            raise EvaluationError("isolated report summary is missing")
        child_result = child_results[0]
        if not isinstance(child_result, Mapping):
            raise EvaluationError("isolated result must be an object")
        if child_result.get("episode") != candidate.episode_dir.name:
            raise EvaluationError("isolated result episode differs from candidate")
        if Path(child_result.get("episode_dir", "")).resolve() != candidate.episode_dir.resolve():
            raise EvaluationError("isolated result path differs from candidate")
        expected_hdf5 = (
            str(candidate.hdf5_path.resolve()) if candidate.hdf5_path else None
        )
        result_hdf5 = child_result.get("hdf5")
        if result_hdf5 is not None:
            result_hdf5 = str(Path(result_hdf5).resolve())
        if result_hdf5 != expected_hdf5:
            raise EvaluationError("isolated result HDF5 differs from candidate")
        if (
            candidate.task_hint is not None
            and child_result.get("task") != candidate.task_hint
        ):
            raise EvaluationError("isolated result task differs from candidate")
        expected_summary = _summarize([dict(child_result)])
        if any(child_summary.get(key) != value for key, value in expected_summary.items()):
            raise EvaluationError("isolated report summary differs from result")
        has_errors = bool(
            expected_summary["error"] or expected_summary["missing_hdf5"]
        )
        expected_status = "complete_with_errors" if has_errors else "complete"
        expected_returncode = 1 if has_errors else 0
        if child.get("status") != expected_status:
            raise EvaluationError("isolated report status differs from result")
        if returncode != expected_returncode:
            raise EvaluationError("isolated evaluator return code differs from report")
    except (OSError, json.JSONDecodeError, EvaluationError) as exc:
        return {
            **base,
            "status": "error",
            "error": f"IsolatedProcessError: {exc}",
            "isolation": isolation,
        }

    result = dict(child_result)
    result["isolation"] = isolation
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("robocasa", "robosuite", "mujoco", "numpy", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _dataset_root_for(input_path: Path) -> Path:
    resolved = input_path.expanduser().resolve(strict=True)
    if resolved.is_file():
        resolved = resolved.parent
    if EPISODE_PATTERN.fullmatch(resolved.name):
        resolved = resolved.parent
    if resolved.name.endswith("_sonic"):
        resolved = resolved.parent
    return resolved


def _default_report_path(dataset_root: Path, run_id: str) -> Path:
    return dataset_root.parent / f"{dataset_root.name}.evaluation-{run_id}.json"


def _build_report(
    *,
    args: argparse.Namespace,
    candidates: Sequence[EpisodeCandidate],
    dataset_root: Path,
    run_id: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "evaluating",
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "dataset_input": str(Path(args.dataset).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "success_mode": args.success_mode,
        "seed": args.seed,
        "selection": {
            "tasks": sorted(args.task),
            "episodes": sorted(args.episode),
            "limit": args.limit,
            "candidate_count": len(candidates),
        },
        "cleanup": {
            "requested": bool(args.delete_failed),
            "mode": "atomic_move_to_recoverable_quarantine",
            "status": "not_started",
        },
        "provenance": {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git": _git_info(repo_root),
        },
        "limitations": [
            "This is state replay: recorded actions are not stepped through controller dynamics.",
            "Raw state_*.npz is preferred because ep_demo.hdf5 drops the true final state to align states with actions.",
            "Python-only task history before recording began is not serialized; in-episode history is reconstructed at recorded post-action cadence.",
            "A terminal state saved between regular post-action checkpoints is explicitly probed once.",
        ],
        "results": [],
    }


def _summarize(results: Sequence[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(results),
        "passed": 0,
        "failed": 0,
        "invalid": 0,
        "error": 0,
        "missing_hdf5": 0,
    }
    for result in results:
        status = result.get("status")
        if status in summary:
            summary[status] += 1
    return summary


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def quarantine_failed_episodes(
    report: dict[str, Any],
    *,
    dataset_root: Path,
    quarantine_root: Path,
    report_path: Path,
) -> None:
    """Atomically move evaluated failures after persisting the complete plan."""

    summary = report["summary"]
    if summary["error"] or summary["missing_hdf5"]:
        report["cleanup"]["status"] = "skipped_due_to_evaluation_errors"
        report["cleanup"]["reason"] = (
            "At least one episode was error or missing_hdf5; no failures were moved."
        )
        _atomic_write_json(report_path, report)
        return

    dataset_root = dataset_root.resolve(strict=True)
    if quarantine_root.exists() or quarantine_root.is_symlink():
        raise EvaluationError(
            f"refusing to reuse quarantine path: {quarantine_root}"
        )
    if quarantine_root.parent.resolve(strict=True) != dataset_root.parent:
        raise EvaluationError("quarantine must be a sibling of the dataset root")
    if os.stat(dataset_root).st_dev != os.stat(dataset_root.parent).st_dev:
        raise EvaluationError("dataset root and quarantine parent differ by filesystem")

    plan: list[dict[str, str]] = []
    seen_targets: set[Path] = set()
    for result in report["results"]:
        if result["status"] not in ("failed", "invalid"):
            continue
        source_input = Path(result["episode_dir"])
        if source_input.is_symlink() or not source_input.is_dir():
            raise EvaluationError(f"failed episode changed type: {source_input}")
        source = source_input.resolve(strict=True)
        if not _is_relative_to(source, dataset_root):
            raise EvaluationError(
                f"refusing to move episode outside dataset root: {source}"
            )
        if not EPISODE_PATTERN.fullmatch(source.name):
            raise EvaluationError(f"unsafe episode directory name: {source}")
        task_name = result.get("task") or source.parent.name.removesuffix("_sonic")
        target = quarantine_root / f"{task_name}_sonic" / source.name
        if target in seen_targets or target.exists() or target.is_symlink():
            raise EvaluationError(f"quarantine target collision: {target}")
        seen_targets.add(target)
        plan.append({"source": str(source), "target": str(target)})

    report["cleanup"].update(
        {
            "status": "planned",
            "quarantine_root": str(quarantine_root),
            "plan": plan,
            "moved": [],
        }
    )
    _atomic_write_json(report_path, report)

    if not plan:
        report["cleanup"]["status"] = "complete_no_failures"
        _atomic_write_json(report_path, report)
        return

    quarantine_root.mkdir(mode=0o755)
    for item in plan:
        source = Path(item["source"])
        target = Path(item["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        report["cleanup"]["moved"].append(item)
        for result in report["results"]:
            if result["episode_dir"] == item["source"]:
                result["quarantined_to"] = item["target"]
                break
        report["cleanup"]["status"] = "moving"
        _atomic_write_json(report_path, report)

    report["cleanup"]["status"] = "complete"
    report["cleanup"]["completed_at"] = dt.datetime.now().astimezone().isoformat()
    _atomic_write_json(report_path, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="sonic_raw root, one Task_sonic directory, one ep_* directory, or one HDF5",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Only evaluate this task name (repeatable).",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Only evaluate this exact ep_* directory name (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many episodes after stable sorting.",
    )
    parser.add_argument(
        "--success-mode",
        choices=("any", "final"),
        default="any",
        help="Pass if any checkpoint succeeds (default), or only the terminal check.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--isolate-episodes",
        action="store_true",
        help=(
            "Evaluate every episode in a fresh Python process so native "
            "simulator corruption cannot affect later episodes."
        ),
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum wall time for each isolated episode process (default: 300).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="External JSON report path (default: sibling of dataset root).",
    )
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help=(
            "Remove fully evaluated failures and deterministically invalid "
            "episodes from the active dataset by moving them to a recoverable "
            "sibling quarantine. Unexpected errors are never moved."
        ),
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Sibling quarantine path used with --delete-failed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    if not (args.episode_timeout_seconds > 0):
        print("error: --episode-timeout-seconds must be positive", file=sys.stderr)
        return 2
    if args.quarantine_dir is not None and not args.delete_failed:
        print("error: --quarantine-dir requires --delete-failed", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_id = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    try:
        dataset_root = _dataset_root_for(args.dataset)
        candidates = discover_episodes(
            args.dataset,
            tasks=args.task,
            episode_names=args.episode,
        )
        if args.limit is not None:
            candidates = candidates[: args.limit]
        report_path = (
            args.report.expanduser().resolve()
            if args.report is not None
            else _default_report_path(dataset_root, run_id)
        )
        if report_path.exists() or report_path.is_symlink():
            raise EvaluationError(f"refusing to overwrite report: {report_path}")
        input_path = args.dataset.expanduser().resolve()
        if input_path.is_dir() and _is_relative_to(report_path, input_path):
            raise EvaluationError(
                f"report must be outside the source dataset: {report_path}"
            )
        for candidate in candidates:
            if _is_relative_to(report_path, candidate.episode_dir.resolve()):
                raise EvaluationError(
                    f"report must not be stored inside an episode: {report_path}"
                )

        report = _build_report(
            args=args,
            candidates=candidates,
            dataset_root=dataset_root,
            run_id=run_id,
        )
        _atomic_write_json(report_path, report)

        child_reports_dir = None
        if args.isolate_episodes:
            child_reports_dir = report_path.parent / (
                f"{report_path.stem}.episodes-{run_id}-{os.getpid()}"
            )
            child_reports_dir.mkdir(mode=0o755)
            report["execution"] = {
                "episode_process_isolation": True,
                "episode_timeout_seconds": args.episode_timeout_seconds,
                "child_reports_dir": str(child_reports_dir),
            }
            _atomic_write_json(report_path, report)

        env_cache = (
            None if args.isolate_episodes else EnvironmentCache(seed=args.seed)
        )
        try:
            for index, candidate in enumerate(candidates, start=1):
                if child_reports_dir is not None:
                    stem = f"{index:06d}-{candidate.episode_dir.name}"
                    result = evaluate_candidate_isolated(
                        candidate,
                        success_mode=args.success_mode,
                        seed=args.seed,
                        timeout_seconds=args.episode_timeout_seconds,
                        report_path=child_reports_dir / f"{stem}.json",
                        log_path=child_reports_dir / f"{stem}.log",
                    )
                else:
                    assert env_cache is not None
                    result = evaluate_candidate(
                        candidate,
                        env_cache,
                        success_mode=args.success_mode,
                    )
                report["results"].append(result)
                report["evaluated_count"] = index
                report["summary"] = _summarize(report["results"])
                _atomic_write_json(report_path, report)
                print(
                    f"[{index}/{len(candidates)}] {result['status'].upper():12s} "
                    f"{result.get('task') or '?'} / {result['episode']}",
                    flush=True,
                )
        finally:
            if env_cache is not None:
                env_cache.close()

        report["summary"] = _summarize(report["results"])
        report["status"] = (
            "complete_with_errors"
            if report["summary"]["error"] or report["summary"]["missing_hdf5"]
            else "complete"
        )
        report["completed_at"] = dt.datetime.now().astimezone().isoformat()
        _atomic_write_json(report_path, report)

        if args.delete_failed:
            quarantine_root = (
                args.quarantine_dir.expanduser().resolve()
                if args.quarantine_dir is not None
                else dataset_root.parent
                / f"{dataset_root.name}.failed-{run_id}"
            )
            quarantine_failed_episodes(
                report,
                dataset_root=dataset_root,
                quarantine_root=quarantine_root,
                report_path=report_path,
            )

    except KeyboardInterrupt:
        if "report" in locals() and "report_path" in locals():
            report["status"] = "interrupted"
            report["completed_at"] = dt.datetime.now().astimezone().isoformat()
            _atomic_write_json(report_path, report)
            print(f"Interrupted. Partial report: {report_path}", file=sys.stderr)
        return 130
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        "Evaluation complete: "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"invalid={summary['invalid']} "
        f"error={summary['error']} missing_hdf5={summary['missing_hdf5']}"
    )
    print(f"Report: {report_path}")
    if args.delete_failed:
        print(f"Cleanup: {report['cleanup']['status']}")
        if report["cleanup"].get("quarantine_root"):
            print(f"Recoverable quarantine: {report['cleanup']['quarantine_root']}")
    return 1 if summary["error"] or summary["missing_hdf5"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
