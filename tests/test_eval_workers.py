"""Policy/simulator lifecycle tests without MuJoCo, model weights, or a GPU."""
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa_eval import gpu_worker
from robocasa_eval.policy import ACTION_DIMS


class FakeSimulator:
    instances = []

    def __init__(self, config, rank, slot, executor):
        self.config, self.rank, self.slot = config, rank, slot
        self.job = self.context = self.observation = self.future = None
        self.start_wall = self.request_wall = 0
        self.last_steps = 0
        self.closed = False
        self.chunks = []
        self.instances.append(self)

    def reset(self, job):
        self.job, self.last_steps = job, 0
        self.call({"command": "reset"})

    def call(self, request):
        self.request_kind = request["command"]
        self.future = Future()
        if self.request_kind == "reset":
            result = {"ok": True, "observation": {"job_id": self.job["id"]}}
        else:
            self.chunks.append(request["actions"])
            self.last_steps += len(request["actions"]["motion_token"])
            finished = self.last_steps >= self.job["max_model_steps"]
            result = dict(ok=True, finished=finished, success=finished,
                          sim_steps=self.last_steps, sim_seconds=self.last_steps / 50,
                          observation={"job_id": self.job["id"]})
        self.future.set_result(result)

    def close(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()
        self.future = None


class TrackingPolicy:
    def __init__(self, namespace="worker-fixture"):
        self.namespace = namespace
        self.active, self.started, self.ended, self.batches = {}, [], [], []
        self.closed = False

    def reset(self, context):
        assert context.episode_id not in self.active
        assert all(other.slot != context.slot for other in self.active.values())
        self.active[context.episode_id] = context
        self.started.append(context)

    def predict_batch(self, observations, contexts):
        for obs, context in zip(observations, contexts):
            assert f"{self.namespace}:{obs['job_id']}" == context.episode_id
            assert self.active[context.episode_id] is context
        self.batches.append(tuple(contexts))
        # Different action lengths per concurrent episode; both shorter than 40.
        return [{key: np.full((1 + context.episode % 2, dim), context.episode, np.float32)
                 for key, dim in ACTION_DIMS.items()} for context in contexts]

    def end_episode(self, context):
        assert self.active.pop(context.episode_id) is context
        self.ended.append(context)

    def close(self):
        assert not self.active
        self.closed = True


@pytest.fixture
def setup_worker(tmp_path, monkeypatch):
    config = dict(output_dir=str(tmp_path / "output"), workspace=str(tmp_path),
                  identity="worker-fixture", policy={}, seed=7,
                  envs_per_gpu=2, model_hz=50, execution_horizon=40,
                  operation_timeout_seconds=10, batch_wait_ms=0)
    jobs = [dict(id=f"Task__ep{i:03}", task=dict(name="Task", category="atomic", split="seen"),
                 rank=0, episode=i, seed=100+i, timeout_seconds=.06,
                 max_model_steps=3) for i in range(4)]
    policy = TrackingPolicy()
    FakeSimulator.instances = []
    monkeypatch.setattr(gpu_worker, "jobs_for", lambda c: jobs)
    monkeypatch.setattr(gpu_worker, "load_policy", lambda c, **kw: policy)
    monkeypatch.setattr(gpu_worker, "Simulator", FakeSimulator)
    monkeypatch.setattr(gpu_worker, "DiskQuotaMonitor", lambda c: SimpleNamespace(
        poll=lambda: None, close=lambda: None))
    return config, jobs, policy


def records(config):
    return [json.loads(path.read_text()) for path in sorted(
        (Path(config["output_dir"]) / "episodes").glob("*.json"))]


def assert_lock_released(config):
    with (Path(config["output_dir"]) / "gpu-00.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_dynamic_batch_preserves_episode_history_and_short_chunks(setup_worker):
    config, jobs, policy = setup_worker
    gpu_worker.run(config, 0, "cpu")
    assert len(policy.started) == len(policy.ended) == 4
    assert policy.closed
    assert any(len(batch) == 2 for batch in policy.batches)
    # At least one batch mixes two episodes that are not the first pair.
    assert any(len(batch) == 2 and {c.episode for c in batch} != {0, 1}
               for batch in policy.batches)
    assert any(len(batch) == 1 for batch in policy.batches)
    assert {row["status"] for row in records(config)} == {"success"}
    assert {row["sim_steps"] for row in records(config)} == {3}
    assert all(slot.closed for slot in FakeSimulator.instances)
    chunks = [chunk for slot in FakeSimulator.instances for chunk in slot.chunks]
    assert {len(chunk["motion_token"]) for chunk in chunks} == {1, 2}
    assert_lock_released(config)


def test_shared_policy_service_episode_ids_are_distinct_across_runs(setup_worker, monkeypatch):
    config, jobs, first = setup_worker
    gpu_worker.run(config, 0, "cpu")
    second = TrackingPolicy(namespace="second-run-fingerprint")
    second_config = dict(config, identity=second.namespace,
                         output_dir=str(Path(config["output_dir"]).with_name("second-output")))
    monkeypatch.setattr(gpu_worker, "load_policy", lambda c, **kw: second)
    gpu_worker.run(second_config, 0, "cpu")
    first_ids = {context.episode_id for context in first.started}
    second_ids = {context.episode_id for context in second.started}
    assert len(first_ids) == len(second_ids) == 4
    assert first_ids.isdisjoint(second_ids)
    assert first_ids == {f"{config['identity']}:{job['id']}" for job in jobs}
    assert second_ids == {f"{second_config['identity']}:{job['id']}" for job in jobs}
    # Resume/result paths retain the existing per-run job identifier convention.
    assert {row["job_id"] for row in records(config)} == {job["id"] for job in jobs}
    assert {row["job_id"] for row in records(second_config)} == {job["id"] for job in jobs}


@pytest.mark.parametrize("invalid", [False, True])
def test_batch_failure_is_terminal_for_affected_episodes(setup_worker, invalid):
    config, jobs, policy = setup_worker
    def bad_batch(observations, contexts):
        policy.batches.append(tuple(contexts))
        if invalid:
            return [{key: np.zeros((0, dim)) for key, dim in ACTION_DIMS.items()}
                    for _ in contexts]
        raise RuntimeError("model server failed")
    policy.predict_batch = bad_batch
    gpu_worker.run(config, 0, "cpu")
    rows = records(config)
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"error"}
    assert all("Policy inference/action contract failed" in row["reason"] for row in rows)
    assert len(policy.batches) == 2  # No implicit retry of failed episodes.
    assert len(policy.started) == len(policy.ended) == 4
    assert policy.closed and all(slot.closed for slot in FakeSimulator.instances)
    assert_lock_released(config)


def test_reset_failure_releases_partial_episode_context(setup_worker):
    config, jobs, policy = setup_worker
    original = policy.reset
    def fail(context):
        original(context)
        raise ValueError("bad episode cache")
    policy.reset = fail
    gpu_worker.run(config, 0, "cpu")
    assert len(records(config)) == 4
    assert {row["status"] for row in records(config)} == {"error"}
    assert len(policy.started) == len(policy.ended) == 4
    assert policy.closed and not policy.batches


def test_end_failure_is_recorded_without_double_cleanup(setup_worker):
    config, jobs, policy = setup_worker
    original = policy.end_episode
    def fail(context):
        original(context)
        raise ValueError("end cleanup failed")
    policy.end_episode = fail
    gpu_worker.run(config, 0, "cpu")
    assert {row["status"] for row in records(config)} == {"error"}
    assert all("end_episode failed" in row["reason"] for row in records(config))
    assert len(policy.ended) == 4 and policy.closed


def test_interrupt_cleans_contexts_but_keeps_inflight_for_resume(setup_worker):
    config, jobs, policy = setup_worker
    original = policy.predict_batch
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(observations, contexts):
        result = original(observations, contexts)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    policy.predict_batch = interrupted
    with pytest.raises(InterruptedError, match="pending for resume"):
        gpu_worker.run(config, 0, "cpu")
    assert not records(config)
    assert len(policy.started) == len(policy.ended) == 2
    assert policy.closed and all(slot.closed for slot in FakeSimulator.instances)
    assert not any(slot.chunks for slot in FakeSimulator.instances)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert_lock_released(config)


def test_fatal_monitor_error_marks_active_episodes_and_closes_all(setup_worker, monkeypatch):
    config, jobs, policy = setup_worker
    def fail():
        raise OSError("disk quota exceeded")
    monkeypatch.setattr(gpu_worker, "DiskQuotaMonitor", lambda c: SimpleNamespace(
        poll=fail, close=lambda: None))
    with pytest.raises(OSError, match="disk quota"):
        gpu_worker.run(config, 0, "cpu")
    assert len(records(config)) == 2
    assert {row["status"] for row in records(config)} == {"error"}
    assert len(policy.started) == len(policy.ended) == 2
    assert policy.closed and all(slot.closed for slot in FakeSimulator.instances)
    assert_lock_released(config)


def test_no_pending_jobs_never_loads_policy_and_releases_lock(setup_worker, monkeypatch):
    config, jobs, policy = setup_worker
    monkeypatch.setattr(gpu_worker, "read_result", lambda *a: {"status": "success"})
    def unexpected(*a, **kw):
        raise AssertionError("Policy must not load")
    monkeypatch.setattr(gpu_worker, "load_policy", unexpected)
    gpu_worker.run(config, 0, "cpu")
    assert not FakeSimulator.instances
    assert_lock_released(config)


def test_load_failure_restores_signal_handlers_and_releases_lock(setup_worker, monkeypatch):
    config, jobs, policy = setup_worker
    previous = signal.getsignal(signal.SIGINT)
    def fail(*a, **kw):
        raise ImportError("adapter unavailable")
    monkeypatch.setattr(gpu_worker, "load_policy", fail)
    with pytest.raises(ImportError, match="adapter unavailable"):
        gpu_worker.run(config, 0, "cpu")
    assert signal.getsignal(signal.SIGINT) == previous
    assert_lock_released(config)


def test_simulator_spawn_failure_closes_its_log(tmp_path, monkeypatch):
    # Inspect the actual class separately from the run-loop fake.
    opened = []
    original = Path.open
    def track(path, *a, **kw):
        stream = original(path, *a, **kw)
        if path.name == "simulator.log":
            opened.append(stream)
        return stream
    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(gpu_worker, "sim_environment", lambda *a: {})
    def fail(*a, **kw):
        assert a[0][-2:] == ["-m", "robocasa_eval.sim_worker"]
        raise OSError("could not spawn simulator")
    monkeypatch.setattr(gpu_worker.subprocess, "Popen", fail)
    config = dict(output_dir=str(tmp_path), workspace=str(tmp_path), sim_python="python")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(OSError, match="could not spawn"):
            gpu_worker.Simulator(config, 0, 0, executor)
    assert len(opened) == 1 and opened[0].closed


def test_simulator_close_terminates_and_waits_for_pending_rpc():
    events = []
    future = Future()
    future.set_running_or_notify_cancel()
    class Process:
        stdin, stdout = io.BytesIO(), io.BytesIO()
        def poll(self):
            return None
        def terminate(self):
            events.append("terminate")
            future.set_exception(EOFError("pipe closed"))
        def wait(self, timeout=None):
            events.append("wait")
    simulator = gpu_worker.Simulator.__new__(gpu_worker.Simulator)
    simulator.proc, simulator.future, simulator.log = Process(), future, io.BytesIO()
    simulator.close()
    assert events == ["terminate", "wait"]
    assert simulator.future is None
    assert simulator.proc.stdin.closed and simulator.proc.stdout.closed and simulator.log.closed
