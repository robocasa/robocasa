"""Thread-owned model replicas behind one policy service port.

Each device owns one policy and one queue. Its thread creates the model, invokes
every lifecycle hook, and closes it. Adjacent prediction requests can be batched;
reset/end commands are ordering barriers. Episode histories stay on the replica
selected by ``context.rank % len(devices)`` for their entire lifetime.

Python cannot interrupt an executing GPU kernel or an arbitrary plugin call.
Request deadlines therefore fail the request and enqueue session cleanup without
retrying inference. Shutdown waits a bounded time and reports any stuck model
thread; daemon threads permit process termination in that situation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import math
import queue
import sys
import threading
import time
from typing import Any

from .policy import EpisodeContext, load_policy, validate_action_batch


class PolicyContextError(ValueError):
    """An episode is unknown, duplicated, busy, or has changed its context."""


@dataclass
class _Command:
    kind: str
    contexts: tuple[EpisodeContext, ...] = ()
    observations: tuple[Mapping[str, Any], ...] = ()
    deadline: float = math.inf
    future: Future = field(default_factory=Future)


@dataclass
class _Session:
    context: EpisodeContext
    replica: int
    state: str = "resetting"
    busy: bool = False
    touched: float = field(default_factory=time.monotonic)
    cleanup: Future | None = None


def _positive_number(config, key, default, *, zero=False):
    value = config.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"{key} must be a finite {'nonnegative' if zero else 'positive'} number.")
    return float(value)


def _positive_integer(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _context(context):
    if not isinstance(context, EpisodeContext):
        raise PolicyContextError("An EpisodeContext is required.")
    for name in ("episode_id", "task"):
        value = getattr(context, name)
        if not isinstance(value, str) or not value.strip():
            raise PolicyContextError(f"context.{name} must be a nonempty string.")
    for name in ("episode", "seed", "rank", "slot"):
        value = getattr(context, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PolicyContextError(f"context.{name} must be a nonnegative integer.")
    return context


def _finish(future, *, result=None, error=None):
    # Only the owner thread completes a command. Callers timing out do not
    # cancel the future, since the corresponding model call may already run.
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)


def _model_error(operation, device, cause):
    error = RuntimeError(f"Policy {operation} failed on {device}.")
    error.__cause__ = cause
    return error


class _Replica:
    def __init__(self, pool, index, device):
        self.pool = pool
        self.index = index
        self.device = device
        self.commands = queue.Queue()
        self.ready = Future()
        self.close_error = None
        self.thread = threading.Thread(
            target=self._run, name=f"policy-{index}-{device}", daemon=True
        )

    def _run(self):
        policy = None
        try:
            policy = self.pool._loader(
                self.pool.config, device=self.device, seed=self.pool.seed + self.index
            )
            _finish(self.ready)
            pending = None
            while True:
                self.pool._collect_idle(self.index)
                if pending is None:
                    try:
                        command = self.commands.get(timeout=min(0.25, self.pool.idle_seconds))
                    except queue.Empty:
                        continue
                else:
                    command, pending = pending, None
                if command.kind == "stop":
                    break
                if command.kind == "end":
                    self._end(policy, command)
                    continue
                if self.pool._closed:
                    self._fail(command, RuntimeError("Policy pool is closing."))
                    continue
                if command.deadline <= time.monotonic():
                    self._fail(command, TimeoutError("Policy request expired before execution."))
                    continue
                if command.kind == "reset":
                    self._reset(policy, command)
                    continue

                batch = [command]
                size = len(command.contexts)
                until = min(command.deadline, time.monotonic() + self.pool.batch_wait_seconds)
                while size < self.pool.max_batch_size:
                    remaining = until - time.monotonic()
                    try:
                        candidate = self.commands.get(timeout=max(0.0, remaining))
                    except queue.Empty:
                        break
                    # A lifecycle command or a request that would exceed the
                    # batch limit remains next in FIFO order.
                    if candidate.kind != "predict" or size + len(candidate.contexts) > self.pool.max_batch_size:
                        pending = candidate
                        break
                    if candidate.deadline <= time.monotonic():
                        self._fail(candidate, TimeoutError("Policy request expired before execution."))
                        continue
                    batch.append(candidate)
                    size += len(candidate.contexts)
                    until = min(until, candidate.deadline)
                self._predict(policy, batch)
        except BaseException as error:
            if not self.ready.done():
                _finish(self.ready, error=_model_error("initialization", self.device, error))
            else:
                self.pool._replica_failed(self.index, error)
        finally:
            if policy is not None:
                try:
                    policy.close()
                except BaseException as error:
                    self.close_error = _model_error("close", self.device, error)

    def _fail(self, command, error):
        self.pool._cleanup_contexts(command.contexts)
        _finish(command.future, error=error)

    def _reset(self, policy, command):
        context = command.contexts[0]
        try:
            policy.reset(context)
        except BaseException as error:
            self._fail(command, _model_error("reset", self.device, error))
            return
        with self.pool._lock:
            session = self.pool._sessions.get(context.episode_id)
            if session is not None and session.state == "resetting":
                session.state = "active"
                session.touched = time.monotonic()
        _finish(command.future)

    def _predict(self, policy, commands):
        active = []
        for command in commands:
            if self.pool._closed or command.deadline <= time.monotonic():
                error = (
                    RuntimeError("Policy pool is closing.")
                    if self.pool._closed
                    else TimeoutError("Policy request expired before execution.")
                )
                self._fail(command, error)
            else:
                active.append(command)
        if not active:
            return
        observations = [obs for command in active for obs in command.observations]
        contexts = [context for command in active for context in command.contexts]
        try:
            actions = policy.predict_batch(observations, contexts)
            # No client execution horizon is applied at the model service.
            actions = validate_action_batch(actions, len(contexts), sys.maxsize)
        except BaseException as error:
            for command in active:
                self._fail(command, _model_error("prediction", self.device, error))
            return
        offset = 0
        for command in active:
            with self.pool._lock:
                for context in command.contexts:
                    session = self.pool._sessions.get(context.episode_id)
                    if session is not None:
                        session.busy = False
                        session.touched = time.monotonic()
            size = len(command.contexts)
            _finish(command.future, result=actions[offset : offset + size])
            offset += size

    def _end(self, policy, command):
        context = command.contexts[0]
        error = None
        try:
            policy.end_episode(context)
        except BaseException as cause:
            error = _model_error("end_episode", self.device, cause)
        with self.pool._lock:
            self.pool._sessions.pop(context.episode_id, None)
            self.pool._completed[context.episode_id] = (context, command.future)
            # Keep recent acknowledgments without growing a long-running
            # service indefinitely. Repeating an unknown end is still a no-op.
            while len(self.pool._completed) > max(1024, self.pool.max_active_episodes * 4):
                self.pool._completed.popitem(last=False)
        _finish(command.future, error=error)


class PolicyPool:
    """Synchronous, thread-safe facade for one or more model replicas.

``config`` supplies workspace/policy, devices, seed, max_batch_size,
batch_wait_ms, request_timeout_seconds, max_active_episodes and
idle_session_seconds. ``loader`` has the same signature as ``load_policy``.
Models initialize sequentially so adapters that set global RNG/cache state do
not race during construction. Each initialized model remains in its own thread.
"""

    def __init__(self, config: Mapping[str, Any], loader=load_policy):
        self.config = dict(config)
        devices = config.get("devices", ["cpu"])
        if (
            not isinstance(devices, (list, tuple))
            or not devices
            or any(not isinstance(device, str) or not device.strip() for device in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ValueError("devices must be a nonempty list of distinct device names.")
        self.devices = tuple(devices)
        self.seed = config.get("seed", 0)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.max_batch_size = _positive_integer(config, "max_batch_size", 32)
        self.max_active_episodes = _positive_integer(config, "max_active_episodes", 4096)
        self.batch_wait_seconds = _positive_number(config, "batch_wait_ms", 5, zero=True) / 1000
        self.request_timeout_seconds = _positive_number(config, "request_timeout_seconds", 120)
        self.idle_seconds = _positive_number(config, "idle_session_seconds", 600)
        self.close_timeout_seconds = _positive_number(
            config, "close_timeout_seconds", min(10.0, self.request_timeout_seconds)
        )
        self._loader = loader
        self._lock = threading.RLock()
        self._sessions = {}
        self._completed = OrderedDict()
        self._closed = False
        self._failed = {}
        self._replicas = []
        try:
            for index, device in enumerate(self.devices):
                replica = _Replica(self, index, device)
                self._replicas.append(replica)
                replica.thread.start()
                replica.ready.result(timeout=self.request_timeout_seconds)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    @property
    def active_episode_count(self):
        with self._lock:
            return len(self._sessions)

    @property
    def ready(self):
        """Whether every configured replica is live and serving requests."""
        with self._lock:
            return (
                not self._closed
                and not self._failed
                and len(self._replicas) == len(self.devices)
                and all(replica.ready.done() and replica.thread.is_alive() for replica in self._replicas)
            )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Policy pool is closed.")

    def _session(self, context):
        session = self._sessions.get(context.episode_id)
        if session is None:
            raise PolicyContextError("Episode is not active; reset it before predicting.")
        if session.context != context:
            raise PolicyContextError("Episode context differs from its reset context.")
        if session.replica in self._failed:
            raise RuntimeError("The episode's model replica has failed.") from self._failed[session.replica]
        return session

    def _cleanup_locked(self, session):
        if session.cleanup is None:
            command = _Command("end", (session.context,))
            session.state = "ending"
            session.cleanup = command.future
            replica = self._replicas[session.replica]
            if session.replica in self._failed:
                self._sessions.pop(session.context.episode_id, None)
                self._completed[session.context.episode_id] = (session.context, command.future)
                _finish(command.future, error=RuntimeError("Model replica failed before session cleanup."))
            else:
                replica.commands.put(command)
        return session.cleanup

    def _cleanup_contexts(self, contexts):
        with self._lock:
            for context in contexts:
                session = self._sessions.get(context.episode_id)
                if session is not None and session.context == context:
                    self._cleanup_locked(session)

    def _collect_idle(self, replica):
        cutoff = time.monotonic() - self.idle_seconds
        with self._lock:
            for session in list(self._sessions.values()):
                if (
                    session.replica == replica
                    and session.state == "active"
                    and not session.busy
                    and session.touched <= cutoff
                ):
                    self._cleanup_locked(session)

    def _replica_failed(self, replica, cause):
        with self._lock:
            self._failed[replica] = cause
            for session in list(self._sessions.values()):
                if session.replica == replica:
                    # The owner thread is no longer available to process its
                    # queued cleanup commands. policy.close() in its finally
                    # block releases the model; complete all lifecycle waiters.
                    future = session.cleanup or Future()
                    if not future.done():
                        _finish(future, error=RuntimeError("Model replica failed before session cleanup."))
                    self._sessions.pop(session.context.episode_id, None)
                    self._completed[session.context.episode_id] = (session.context, future)
            commands = self._replicas[replica].commands
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if not command.future.done():
                    _finish(command.future, error=RuntimeError("Model replica has stopped."))

    def _wait(self, future, deadline, contexts=()):
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError as error:
            self._cleanup_contexts(contexts)
            raise TimeoutError("Policy request timed out; its episodes cannot continue.") from error

    def reset(self, context: EpisodeContext) -> None:
        context = _context(context)
        deadline = time.monotonic() + self.request_timeout_seconds
        with self._lock:
            self._ensure_open()
            if context.episode_id in self._sessions or context.episode_id in self._completed:
                raise PolicyContextError("Episode ID has already been reset; use a new episode ID.")
            if len(self._sessions) >= self.max_active_episodes:
                raise RuntimeError("The policy service has reached max_active_episodes.")
            replica = context.rank % len(self.devices)
            if replica in self._failed:
                raise RuntimeError("The requested model replica has failed.") from self._failed[replica]
            session = _Session(context, replica)
            self._sessions[context.episode_id] = session
            command = _Command("reset", (context,), deadline=deadline)
            self._replicas[replica].commands.put(command)
        self._wait(command.future, deadline, (context,))

    def predict_batch(self, observations, contexts) -> list[dict[str, Any]]:
        if (
            not isinstance(observations, Sequence)
            or isinstance(observations, (str, bytes))
            or not isinstance(contexts, Sequence)
            or isinstance(contexts, (str, bytes))
            or not contexts
            or len(observations) != len(contexts)
            or any(not isinstance(observation, Mapping) for observation in observations)
        ):
            raise ValueError("predict_batch needs equally sized, nonempty observation and context lists.")
        contexts = tuple(_context(context) for context in contexts)
        if len({context.episode_id for context in contexts}) != len(contexts):
            raise PolicyContextError("An episode can appear only once in a prediction batch.")
        deadline = time.monotonic() + self.request_timeout_seconds
        commands = []
        with self._lock:
            self._ensure_open()
            sessions = [self._session(context) for context in contexts]
            if any(session.state != "active" or session.busy for session in sessions):
                raise PolicyContextError("Episode is resetting, ending, or already has a prediction in flight.")
            grouped = {}
            for index, session in enumerate(sessions):
                session.busy = True
                session.touched = time.monotonic()
                grouped.setdefault(session.replica, []).append(index)
            for replica, indices in grouped.items():
                for start in range(0, len(indices), self.max_batch_size):
                    selected = indices[start : start + self.max_batch_size]
                    command = _Command(
                        "predict",
                        tuple(contexts[index] for index in selected),
                        tuple(observations[index] for index in selected),
                        deadline,
                    )
                    self._replicas[replica].commands.put(command)
                    commands.append((selected, command))
        results = [None] * len(contexts)
        try:
            for indices, command in commands:
                actions = self._wait(command.future, deadline, contexts)
                for index, action in zip(indices, actions):
                    results[index] = action
        except BaseException:
            # Work already executing cannot be interrupted. Requests that have
            # not started on another replica must not run after this aggregate
            # request has failed.
            with self._lock:
                for _, command in commands:
                    command.deadline = min(command.deadline, time.monotonic())
            self._cleanup_contexts(contexts)
            raise
        return results

    def end_episode(self, context: EpisodeContext) -> None:
        context = _context(context)
        deadline = time.monotonic() + self.request_timeout_seconds
        with self._lock:
            completed = self._completed.get(context.episode_id)
            if completed is not None:
                if completed[0] != context:
                    raise PolicyContextError("Episode context differs from its reset context.")
                future = completed[1]
            else:
                session = self._sessions.get(context.episode_id)
                if session is None:
                    # Cleanup is idempotent even after the bounded completion
                    # acknowledgment cache expires or a reset never arrived.
                    return
                if session.context != context:
                    raise PolicyContextError("Episode context differs from its reset context.")
                future = self._cleanup_locked(session)
        self._wait(future, deadline)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                for session in list(self._sessions.values()):
                    self._cleanup_locked(session)
                for replica in self._replicas:
                    replica.commands.put(_Command("stop"))
        deadline = time.monotonic() + self.close_timeout_seconds
        for replica in self._replicas:
            replica.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        stuck = [replica.device for replica in self._replicas if replica.thread.is_alive()]
        if stuck:
            raise TimeoutError(f"Policy calls are still running during shutdown on {', '.join(stuck)}.")
        errors = [replica.close_error for replica in self._replicas if replica.close_error is not None]
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
