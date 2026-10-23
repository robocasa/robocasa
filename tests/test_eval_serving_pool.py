"""CPU coverage of the single-port service's thread-owned model replicas."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from robocasa_eval.policy import ACTION_DIMS, EpisodeContext
from robocasa_eval.serving_pool import PolicyContextError, PolicyPool


def context(index=0, *, rank=0):
    return EpisodeContext(f"episode-{index}", "Task", index, 100 + index, rank, index)


class FakePolicy:
    def __init__(self, *, device, log, fail=None, entered=None, release=None):
        self.device = device
        self.log = log
        self.fail = fail
        self.entered = entered
        self.release = release
        self.owner = threading.get_ident()
        self.sessions = {}
        self.closed = False
        self._log("load", ())

    def _log(self, event, contexts):
        assert threading.get_ident() == self.owner
        self.log.append((self.device, event, tuple(item.episode_id for item in contexts), self.owner))

    def reset(self, item):
        self._log("reset", (item,))
        assert item.episode_id not in self.sessions
        self.sessions[item.episode_id] = 0
        if self.fail == "reset":
            raise RuntimeError("fake reset failure")

    def predict_batch(self, observations, contexts):
        self._log("predict", contexts)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=3)
        if self.fail == "predict":
            raise RuntimeError("fake inference failure")
        if self.fail == "bad_action":
            return [{"motion_token": np.zeros((1, 64))} for _ in contexts]
        result = []
        for obs, item in zip(observations, contexts):
            assert item.episode_id in self.sessions
            self.sessions[item.episode_id] += 1
            value = obs.get("value", item.episode) + 1000 * self.sessions[item.episode_id]
            result.append({key: np.full((3, dim), value) for key, dim in ACTION_DIMS.items()})
        return result

    def end_episode(self, item):
        self._log("end", (item,))
        self.sessions.pop(item.episode_id, None)
        if self.fail == "end":
            raise RuntimeError("fake cleanup failure")

    def close(self):
        self._log("close", ())
        self.closed = True
        self.sessions.clear()
        if self.fail == "close":
            raise RuntimeError("fake close failure")


@pytest.fixture
def make_pool():
    resources = []

    def make(*, devices=None, fake_options=None, **options):
        log = []
        models = []
        fake_options = fake_options or {}

        def loader(config, *, device, seed):
            model = FakePolicy(device=device, log=log, **fake_options)
            models.append(model)
            return model

        config = {
            "workspace": "unused",
            "policy": {},
            "devices": devices or ["cpu"],
            "seed": 7,
            "request_timeout_seconds": 1,
            **options,
        }
        pool = PolicyPool(config, loader=loader)
        resources.append((pool, fake_options))
        return pool, models, log

    yield make
    for pool, options in resources:
        if options.get("release") is not None:
            options["release"].set()
        try:
            pool.close()
        except RuntimeError:
            pass  # close-error behavior has its own assertion.


def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_replicas_own_every_call_and_route_stably(make_pool):
    pool, models, log = make_pool(devices=["cpu:0", "cpu:1"])
    episodes = [context(index, rank=index) for index in range(4)]
    for item in episodes:
        pool.reset(item)
    order = [episodes[3], episodes[0], episodes[2], episodes[1]]
    actions = pool.predict_batch([{"value": item.episode} for item in order], order)
    assert [float(item["motion_token"][0, 0]) for item in actions] == [1003, 1000, 1002, 1001]
    for item in episodes:
        pool.end_episode(item)
    pool.close()
    assert all(model.closed for model in models)
    assert len({model.owner for model in models}) == 2
    for device, event, ids, owner in log:
        if event in ("reset", "predict", "end"):
            assert all(int(identity.split("-")[-1]) % 2 == int(device[-1]) for identity in ids)
        assert owner != threading.get_ident()


def test_model_construction_is_sequential_and_thread_owned():
    log = []
    loading = threading.Lock()

    def loader(config, *, device, seed):
        assert loading.acquire(blocking=False)
        try:
            time.sleep(0.005)
            return FakePolicy(device=device, log=log)
        finally:
            loading.release()

    with PolicyPool({"devices": ["cpu:0", "cpu:1", "cpu:2"]}, loader=loader):
        assert [entry[0] for entry in log if entry[1] == "load"] == ["cpu:0", "cpu:1", "cpu:2"]


def test_batches_near_simultaneous_requests_and_keeps_histories(make_pool):
    pool, models, log = make_pool(batch_wait_ms=100, max_batch_size=8)
    episodes = [context(index) for index in range(4)]
    for item in episodes:
        pool.reset(item)
    barrier = threading.Barrier(len(episodes))

    def predict(item):
        barrier.wait(timeout=2)
        return pool.predict_batch([{"value": item.episode}], [item])[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        actions = list(executor.map(predict, episodes))
    batches = [entry[2] for entry in log if entry[1] == "predict"]
    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert [item["motion_token"][0, 0] for item in actions] == [1000, 1001, 1002, 1003]
    later = pool.predict_batch([{}, {}], [episodes[3], episodes[0]])
    assert [item["motion_token"][0, 0] for item in later] == [2003, 2000]


def test_large_request_splits_at_batch_limit_and_preserves_order(make_pool):
    pool, _, log = make_pool(max_batch_size=2)
    episodes = [context(index) for index in range(5)]
    for item in episodes:
        pool.reset(item)
    actions = pool.predict_batch([{} for _ in episodes], episodes)
    assert [item["motion_token"][0, 0] for item in actions] == [1000, 1001, 1002, 1003, 1004]
    assert [len(item[2]) for item in log if item[1] == "predict"] == [2, 2, 1]
    assert all(item["motion_token"].shape == (3, 64) for item in actions)


def test_lifecycle_commands_are_batch_barriers(make_pool):
    pool, _, log = make_pool(batch_wait_ms=150)
    first, second = context(1), context(2)
    pool.reset(first)
    with ThreadPoolExecutor(max_workers=2) as executor:
        predicted = executor.submit(pool.predict_batch, [{}], [first])
        eventually(lambda: pool._sessions[first.episode_id].busy)
        pool.reset(second)
        pool.predict_batch([{}], [second])
        predicted.result(timeout=2)
    events = [(event, ids) for _, event, ids, _ in log if event not in ("load", "close")]
    assert events == [
        ("reset", (first.episode_id,)),
        ("predict", (first.episode_id,)),
        ("reset", (second.episode_id,)),
        ("predict", (second.episode_id,)),
    ]


def test_context_reuse_mutation_unknown_and_duplicate_batch_rejected(make_pool):
    pool, _, _ = make_pool()
    item = context()
    pool.reset(item)
    with pytest.raises(PolicyContextError, match="already been reset"):
        pool.reset(item)
    with pytest.raises(PolicyContextError, match="differs"):
        pool.predict_batch([{}], [replace(item, task="Other")])
    with pytest.raises(PolicyContextError, match="differs"):
        pool.end_episode(replace(item, seed=999))
    with pytest.raises(PolicyContextError, match="only once"):
        pool.predict_batch([{}, {}], [item, item])
    with pytest.raises(PolicyContextError, match="not active"):
        pool.predict_batch([{}], [context(9)])
    pool.end_episode(item)
    pool.end_episode(item)
    pool.end_episode(context(99))
    with pytest.raises(PolicyContextError, match="already been reset"):
        pool.reset(item)
    with pytest.raises(PolicyContextError, match="differs"):
        pool.end_episode(replace(item, task="Other"))


def test_same_episode_cannot_have_overlapping_predictions(make_pool):
    entered, release = threading.Event(), threading.Event()
    pool, _, log = make_pool(fake_options={"entered": entered, "release": release})
    item = context()
    pool.reset(item)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(pool.predict_batch, [{}], [item])
        assert entered.wait(timeout=1)
        with pytest.raises(PolicyContextError, match="in flight"):
            pool.predict_batch([{}], [item])
        ending = executor.submit(pool.end_episode, item)
        assert not ending.done()
        release.set()
        pending.result(timeout=1)
        ending.result(timeout=1)
    assert [item[1] for item in log] == ["load", "reset", "predict", "end"]


@pytest.mark.parametrize("failure", ["reset", "predict", "bad_action"])
def test_failure_cleans_session_without_retry(make_pool, failure):
    pool, models, log = make_pool(fake_options={"fail": failure})
    item = context()
    if failure == "reset":
        with pytest.raises(RuntimeError, match="reset failed"):
            pool.reset(item)
    else:
        pool.reset(item)
        with pytest.raises(RuntimeError, match="prediction failed"):
            pool.predict_batch([{}], [item])
    eventually(lambda: pool.active_episode_count == 0)
    assert not models[0].sessions
    assert sum(entry[1] == "end" for entry in log) == 1
    assert sum(entry[1] == "predict" for entry in log) == (failure != "reset")


def test_failed_coalesced_request_cleans_every_episode(make_pool):
    pool, _, log = make_pool(fake_options={"fail": "predict"}, batch_wait_ms=100)
    episodes = [context(index) for index in range(3)]
    for item in episodes:
        pool.reset(item)
    barrier = threading.Barrier(3)

    def predict(item):
        barrier.wait(timeout=1)
        with pytest.raises(RuntimeError, match="prediction failed"):
            pool.predict_batch([{}], [item])

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(predict, episodes))
    eventually(lambda: pool.active_episode_count == 0)
    assert sum(entry[1] == "predict" for entry in log) == 1
    assert sum(entry[1] == "end" for entry in log) == 3


def test_timeout_fails_episode_and_cleanup_follows_running_call(make_pool):
    entered, release = threading.Event(), threading.Event()
    pool, models, log = make_pool(
        fake_options={"entered": entered, "release": release}, request_timeout_seconds=0.05
    )
    item = context()
    pool.reset(item)
    with pytest.raises(TimeoutError, match="cannot continue"):
        pool.predict_batch([{}], [item])
    assert entered.is_set()
    with pytest.raises(PolicyContextError):
        pool.predict_batch([{}], [item])
    release.set()
    eventually(lambda: pool.active_episode_count == 0)
    assert not models[0].sessions
    assert [entry[1] for entry in log] == ["load", "reset", "predict", "end"]


def test_expired_queued_prediction_is_not_executed(make_pool):
    entered, release = threading.Event(), threading.Event()
    pool, _, log = make_pool(
        fake_options={"entered": entered, "release": release},
        request_timeout_seconds=0.06,
        max_batch_size=1,
    )
    first, second = context(1), context(2)
    pool.reset(first)
    pool.reset(second)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(pool.predict_batch, [{}], [first])
        assert entered.wait(timeout=1)
        with pytest.raises(TimeoutError):
            pool.predict_batch([{}], [second])
        with pytest.raises(TimeoutError):
            pending.result(timeout=1)
        release.set()
    eventually(lambda: pool.active_episode_count == 0)
    assert [entry[2] for entry in log if entry[1] == "predict"] == [(first.episode_id,)]


def test_idle_session_cleanup_releases_capacity(make_pool):
    pool, models, log = make_pool(idle_session_seconds=0.04, max_active_episodes=1)
    first, second = context(1), context(2)
    pool.reset(first)
    with pytest.raises(RuntimeError, match="max_active_episodes"):
        pool.reset(second)
    eventually(lambda: pool.active_episode_count == 0)
    assert not models[0].sessions
    pool.end_episode(first)
    pool.reset(second)
    assert pool.active_episode_count == 1
    assert sum(entry[1] == "end" for entry in log) == 1


def test_cleanup_failure_is_reported_again_for_lost_ack(make_pool):
    pool, _, log = make_pool(fake_options={"fail": "end"})
    item = context()
    pool.reset(item)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="end_episode failed"):
            pool.end_episode(item)
    assert pool.active_episode_count == 0
    assert sum(entry[1] == "end" for entry in log) == 1


def test_shutdown_cleans_live_sessions_then_closes_models(make_pool):
    pool, models, log = make_pool(devices=["cpu:0", "cpu:1"])
    assert pool.ready
    pool.reset(context(0, rank=0))
    pool.reset(context(1, rank=1))
    pool.close()
    pool.close()
    assert not pool.ready
    assert pool.active_episode_count == 0
    assert all(model.closed for model in models)
    for device in pool.devices:
        assert [entry[1] for entry in log if entry[0] == device] == ["load", "reset", "end", "close"]
    with pytest.raises(RuntimeError, match="closed"):
        pool.reset(context(9))


def test_shutdown_is_bounded_for_uninterruptible_plugin(make_pool):
    entered, release = threading.Event(), threading.Event()
    pool, models, _ = make_pool(
        fake_options={"entered": entered, "release": release},
        request_timeout_seconds=0.05,
        close_timeout_seconds=0.02,
    )
    item = context()
    pool.reset(item)
    with pytest.raises(TimeoutError):
        pool.predict_batch([{}], [item])
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="still running"):
        pool.close()
    assert time.monotonic() - started < 0.5
    release.set()
    eventually(lambda: models[0].closed)
    pool.close()


def test_initialization_failure_closes_previous_replica():
    models, log = [], []

    def loader(config, *, device, seed):
        if device == "cpu:1":
            raise RuntimeError("fake loading failure")
        model = FakePolicy(device=device, log=log)
        models.append(model)
        return model

    with pytest.raises(RuntimeError, match="initialization failed"):
        PolicyPool({"devices": ["cpu:0", "cpu:1"]}, loader=loader)
    assert models[0].closed


def test_resume_attempt_id_keeps_same_job_seed_and_fresh_model_history(make_pool):
    pool, _, log = make_pool()
    first = replace(context(), episode_id="job-0:attempt-a")
    resumed = replace(first, episode_id="job-0:attempt-b")
    pool.reset(first)
    old_action = pool.predict_batch([{}], [first])[0]
    pool.end_episode(first)
    pool.reset(resumed)
    pool.end_episode(first)  # A late cleanup acknowledgment cannot end resumed.
    new_action = pool.predict_batch([{}], [resumed])[0]
    assert new_action["motion_token"][0, 0] == old_action["motion_token"][0, 0] == 1000
    with pytest.raises(PolicyContextError):
        pool.predict_batch([{}], [first])
    assert pool.active_episode_count == 1
    assert [entry[2] for entry in log if entry[1] == "end"] == [(first.episode_id,)]


def test_concurrent_duplicate_end_is_single_model_cleanup(make_pool):
    pool, _, log = make_pool()
    item = context()
    pool.reset(item)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(pool.end_episode, [item] * 8))
    assert sum(entry[1] == "end" for entry in log) == 1


def test_internal_replica_failure_clears_sessions_and_marks_unready(make_pool, monkeypatch):
    pool, models, _ = make_pool()
    item = context()
    pool.reset(item)

    def fail_idle(replica):
        raise RuntimeError("unexpected scheduling failure")

    monkeypatch.setattr(pool, "_collect_idle", fail_idle)
    eventually(lambda: models[0].closed)
    assert not pool.ready
    assert pool.active_episode_count == 0
    with pytest.raises(RuntimeError, match="failed"):
        pool.end_episode(item)
    with pytest.raises(RuntimeError, match="failed"):
        pool.reset(context(99))


def test_close_failure_is_reported(make_pool):
    pool, _, _ = make_pool(fake_options={"fail": "close"})
    with pytest.raises(RuntimeError, match="close failed"):
        pool.close()


@pytest.mark.parametrize(
    "key,value",
    [
        ("devices", []),
        ("devices", ["cpu", "cpu"]),
        ("devices", "cuda:0"),
        ("devices", [None]),
        ("max_batch_size", 0),
        ("max_batch_size", True),
        ("max_active_episodes", 1.5),
        ("request_timeout_seconds", float("nan")),
        ("idle_session_seconds", 0),
        ("batch_wait_ms", -1),
        ("seed", -1),
    ],
)
def test_invalid_pool_options_fail_before_loading(key, value):
    def loader(*args, **kwargs):
        raise AssertionError("No model should be loaded")

    with pytest.raises(ValueError):
        PolicyPool({key: value}, loader=loader)


@pytest.mark.parametrize(
    "item",
    [None, replace(context(), rank=-1), replace(context(), seed=True), replace(context(), task="")],
)
def test_invalid_context_is_rejected(make_pool, item):
    pool, _, _ = make_pool()
    with pytest.raises(PolicyContextError):
        pool.reset(item)


@pytest.mark.parametrize("observations,contexts", [([], []), ([{}], []), ([None], [context()]), ("bad", [context()])])
def test_invalid_prediction_envelope_is_rejected(make_pool, observations, contexts):
    pool, _, _ = make_pool()
    with pytest.raises(ValueError):
        pool.predict_batch(observations, contexts)
