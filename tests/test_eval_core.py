"""CPU regression checks for benchmark scheduling, resume, quotas, and scoring."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import pytest
from robocasa_eval import evaluation, gpu_worker

@pytest.fixture
def config(tmp_path):
    workspace = tmp_path / "workspace"
    return {
        "workspace": str(workspace),
        "output_dir": str(workspace / "eval_results" / "test-evaluation"),
        "tasks": evaluation.load_tasks()["tasks"],
        "episodes_per_task": 30,
        "seed": 20261020,
        "atomic_seconds": 40,
        "composite_seconds": 180,
        "model_hz": 50,
        "total_gpus": 16,
        "identity": "fake-fixture-identity",
    }


def write_result(config, job, status, **kwargs):
    evaluation.atomic_json(
        evaluation.result_path(config["output_dir"], job),
        {
            "identity": config["identity"],
            "job_id": job["id"],
            "status": status,
            "sim_seconds": 1.0,
            "wall_seconds": 2.0,
            **kwargs,
        },
    )


def test_atomic_json_concurrent_container_pids_do_not_share_tempfile(tmp_path, monkeypatch):
    """Different containers may have the same PID while sharing the output mount."""
    target = tmp_path / "summary.json"
    barrier = threading.Barrier(2)
    temporary_paths = []
    replace = os.replace
    monkeypatch.setattr(evaluation.os, "getpid", lambda: 17)

    def synchronized_replace(source, destination):
        if Path(destination) == target:
            temporary_paths.append(Path(source))
            barrier.wait(timeout=5)
        return replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", synchronized_replace)
    payloads = [{"node": node, "rows": list(range(node, node + 500))} for node in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluation.atomic_json, target, value) for value in payloads]
        for future in futures:
            future.result(timeout=10)
    assert len(set(temporary_paths)) == 2
    assert json.loads(target.read_text()) in payloads
    assert not list(tmp_path.glob(".summary.json.*.tmp"))


def test_simultaneous_node_summaries_with_same_pid_are_atomic(config, monkeypatch):
    """Synchronize callers before the shared read/write lock, never inside it."""
    config["tasks"] = config["tasks"][:2]
    config["episodes_per_task"] = 2
    output = Path(config["output_dir"])
    targets = {output / name for name in ("summary.json", "episodes.csv", "tasks.csv")}
    temporary_paths = {path: [] for path in targets}
    start = threading.Barrier(2)
    replace = os.replace
    monkeypatch.setattr(evaluation.os, "getpid", lambda: 23)

    def synchronized_replace(source, destination):
        destination = Path(destination)
        if destination in targets:
            temporary_paths[destination].append(Path(source))
        return replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", synchronized_replace)

    def simultaneous_summary():
        start.wait(timeout=5)
        return evaluation.summarize(config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(simultaneous_summary) for _ in range(2)]
        reports = [future.result(timeout=10) for future in futures]
    assert all(report["groups"]["all"]["planned"] == 4 for report in reports)
    assert all(len(set(paths)) == 2 for paths in temporary_paths.values())
    assert json.loads((output / "summary.json").read_text())["groups"]["all"]["planned"] == 4
    assert len((output / "episodes.csv").read_text().splitlines()) == 5
    assert len((output / "tasks.csv").read_text().splitlines()) == 3
    assert not list(output.glob(".*.tmp"))


def test_delayed_pending_summary_cannot_overwrite_newer_complete_summary(config, monkeypatch):
    """Pause an old snapshot while another node observes the final episode arriving."""
    config["tasks"] = config["tasks"][:1]
    config["episodes_per_task"] = 1
    job = evaluation.jobs_for(config)[0]
    output = Path(config["output_dir"])
    old_snapshot_ready = threading.Event()
    release_old_snapshot = threading.Event()
    new_lock_attempt = threading.Event()
    new_read_started = threading.Event()
    role = threading.local()
    writes = []
    atomic_json = evaluation.atomic_json
    read_result = evaluation.read_result
    flock = evaluation.fcntl.flock

    def delayed_write(path, value):
        if Path(path) == output / "summary.json":
            if getattr(role, "name", None) == "old":
                assert value["groups"]["all"]["pending"] == 1
                old_snapshot_ready.set()
                assert release_old_snapshot.wait(timeout=5)
            writes.append(value["groups"]["all"]["pending"])
        return atomic_json(path, value)

    def observe_read(*args):
        if getattr(role, "name", None) == "new":
            new_read_started.set()
        return read_result(*args)

    def observe_lock(fd, operation):
        if getattr(role, "name", None) == "new":
            new_lock_attempt.set()
        return flock(fd, operation)

    def node_summary(name):
        role.name = name
        return evaluation.summarize(config)

    monkeypatch.setattr(evaluation, "atomic_json", delayed_write)
    monkeypatch.setattr(evaluation, "read_result", observe_read)
    monkeypatch.setattr(evaluation.fcntl, "flock", observe_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(node_summary, "old")
        try:
            assert old_snapshot_ready.wait(timeout=5)
            write_result(config, job, "success")
            new = executor.submit(node_summary, "new")
            assert new_lock_attempt.wait(timeout=5)
            assert not new_read_started.is_set(), "The lock must cover reading the snapshot"
        finally:
            release_old_snapshot.set()
        assert old.result(timeout=5)["complete"] is False
        assert new.result(timeout=5)["complete"] is True
    assert writes == [1, 0]
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    import csv
    with (output / "episodes.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "success"
    with (output / "tasks.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["pending"] == "0"


def test_read_only_summary_needs_no_lock_or_output_directory(config, monkeypatch):
    def unexpected_lock(*args):
        raise AssertionError("A read-only summary must not acquire the write lock")

    monkeypatch.setattr(evaluation.fcntl, "flock", unexpected_lock)
    report = evaluation.summarize(config, write=False)
    assert report["groups"]["all"]["pending"] == 1500
    assert not Path(config["output_dir"]).exists()


def test_complete_1500_episode_plan_and_exact_limits(config):
    jobs = evaluation.jobs_for(config)
    assert len(jobs) == 1500
    assert len({job["id"] for job in jobs}) == 1500
    assert len({(job["task"]["name"], job["episode"]) for job in jobs}) == 1500
    assert Counter(job["task"]["name"] for job in jobs) == {
        task["name"]: 30 for task in config["tasks"]
    }
    for task in config["tasks"]:
        assert {job["episode"] for job in jobs if job["task"]["name"] == task["name"]} == set(range(30))
    assert Counter(job["task"]["split"] for job in jobs) == {"seen": 1200, "unseen": 300}
    assert Counter(job["task"]["category"] for job in jobs) == {"atomic": 750, "composite": 750}
    for job in jobs:
        expected_seconds, expected_steps = (
            (40, 2000) if job["task"]["category"] == "atomic" else (180, 9000)
        )
        assert job["timeout_seconds"] == expected_seconds
        assert job["max_model_steps"] == expected_steps
        assert 0 <= job["seed"] < 2**32
        assert 0 <= job["rank"] < 16


def test_seed_stability_independent_of_rng_task_order_and_gpu_count(config):
    first = {job["id"]: job["seed"] for job in evaluation.jobs_for(config)}
    changed = copy.deepcopy(config)
    random.seed(17)
    random.shuffle(changed["tasks"])
    changed["total_gpus"] = 1
    assert {job["id"]: job["seed"] for job in evaluation.jobs_for(changed)} == first
    changed["seed"] += 1
    second = {job["id"]: job["seed"] for job in evaluation.jobs_for(changed)}
    assert all(second[key] != first[key] for key in first)


def test_weighted_rank_allocation_and_coverage(config):
    jobs = evaluation.jobs_for(config)
    by_rank = {rank: [j for j in jobs if j["rank"] == rank] for rank in range(16)}
    assert all(by_rank.values())
    assert len({job["id"] for rows in by_rank.values() for job in rows}) == 1500
    loads = [sum(j["timeout_seconds"] for j in rows) for rows in by_rank.values()]
    assert sum(loads) == 750 * 40 + 750 * 180
    # A rank's worst-case simulation budget must stay within one long episode.
    assert max(loads) - min(loads) <= 180
    for rows in by_rank.values():
        assert {j["task"]["category"] for j in rows} == {"atomic", "composite"}


def test_default_single_node_one_worker_topology():
    assert evaluation.topology({}) == {
        "total": 1, "nodes": 1, "node_rank": 0, "local": 1,
        "selected": [0],
    }


def test_cpu_budget_respects_v1_quota_under_large_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.os, "sched_getaffinity", lambda _: set(range(128)))
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("800000")
    (tmp_path / "cpu/cpu.cfs_period_us").write_text("100000")
    assert evaluation.available_cpu_cores(tmp_path) == 8


def test_cpu_budget_respects_v2_quota_and_unlimited_value(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.os, "sched_getaffinity", lambda _: set(range(16)))
    (tmp_path / "cpu.max").write_text("400000 100000")
    assert evaluation.available_cpu_cores(tmp_path) == 4
    (tmp_path / "cpu.max").write_text("max 100000")
    assert evaluation.available_cpu_cores(tmp_path) == 16


def test_disk_accounting_handles_disappearing_cache_only(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    result = subprocess.CompletedProcess(
        ["du"], 1, stdout=f"123456\t{root}\n",
        stderr=f"du: cannot access '{root}/cache/uv/.tmp123': No such file or directory\n")
    class FinishedProcess:
        def __init__(self):
            self.returncode = result.returncode
            self.stdout, self.stderr = io.StringIO(), io.StringIO()

        def communicate(self, timeout=None):
            return result.stdout, result.stderr

        def poll(self):
            return self.returncode

    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **kw: FinishedProcess())
    assert evaluation.directory_usage(root) == 123456
    result.stderr = f"du: cannot read directory '{root}/cache/protected': Permission denied\n"
    with pytest.raises(subprocess.CalledProcessError):
        evaluation.directory_usage(root)
    result.stderr = f"du: cannot access '{root}/runs/model': No such file or directory\n"
    with pytest.raises(subprocess.CalledProcessError):
        evaluation.directory_usage(root)


def test_periodic_disk_check_skips_initial_scan_and_never_overlaps():
    now = [1000.0]
    started, release = threading.Event(), threading.Event()
    calls = []

    def check(config, cancel_event):
        calls.append(config)
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("The main loop blocked while a disk scan was running")
        return {"user_directory_bytes": 123}

    monitor = gpu_worker.DiskQuotaMonitor({}, check=check, clock=lambda: now[0])
    try:
        monitor.poll()
        assert monitor.future is None
        now[0] += 599
        monitor.poll()
        assert monitor.future is None
        now[0] += 1
        monitor.poll()
        future = monitor.future
        assert started.wait(timeout=5)
        for _ in range(100):
            now[0] += 1000
            monitor.poll()
            assert monitor.future is future
        assert len(calls) == 1
        release.set()
        future.result(timeout=5)
        monitor.poll()
        assert monitor.future is None
        now[0] += 599
        monitor.poll()
        assert monitor.future is None
        now[0] += 1
        monitor.poll()
        monitor.future.result(timeout=5)
        assert len(calls) == 2
    finally:
        release.set()
        monitor.close()


@pytest.mark.parametrize("failure", [ValueError("quota exceeded"), OSError("disk scan failed")])
def test_periodic_disk_check_propagates_completed_failure(failure):
    now = [0.0]

    def check(config, cancel_event):
        raise failure

    monitor = gpu_worker.DiskQuotaMonitor({}, check=check, clock=lambda: now[0])
    try:
        now[0] = 600
        monitor.poll()
        with pytest.raises(type(failure), match=str(failure)):
            monitor.future.result(timeout=5)
        with pytest.raises(type(failure), match=str(failure)):
            monitor.poll()
        assert monitor.future is None
    finally:
        monitor.close()


def test_disk_monitor_close_does_not_hide_finished_error():
    now = [0.0]

    def check(config, cancel_event):
        raise ValueError("quota exceeded at completion")

    monitor = gpu_worker.DiskQuotaMonitor({}, check=check, clock=lambda: now[0])
    now[0] = 600
    monitor.poll()
    with pytest.raises(ValueError, match="quota exceeded"):
        monitor.future.result(timeout=5)
    with pytest.raises(ValueError, match="quota exceeded"):
        monitor.close()
    monitor.close()  # Cleanup remains safe after the error was reported.


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_disk_monitor_cancel_reaps_running_du(tmp_path, monkeypatch, ignore_terminate):
    """Use a real process so cancellation also covers terminate/kill/wait behavior."""
    ready = tmp_path / "du-ready"
    code = (
        "import signal,time\n"
        "from pathlib import Path\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_terminate else "")
        + f"Path({str(ready)!r}).write_text('ready')\n"
        + "time.sleep(60)\n"
    )
    processes = []
    popen = subprocess.Popen

    def fake_du(command, **kwargs):
        assert command[:3] == ["du", "-s", "-B1"]
        process = popen([sys.executable, "-B", "-c", code], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(evaluation.subprocess, "Popen", fake_du)
    now = [0.0]

    def check(config, cancel_event):
        return evaluation.directory_usage(tmp_path, cancel_event=cancel_event)

    monitor = gpu_worker.DiskQuotaMonitor({}, check=check, clock=lambda: now[0])
    try:
        now[0] = 600
        monitor.poll()
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline, "The controlled du process did not start"
            time.sleep(0.01)
        assert len(processes) == 1 and processes[0].poll() is None
        future = monitor.future
        monitor.close()
        assert isinstance(future.exception(), evaluation.DiskCheckCancelled)
        assert processes[0].returncode == (-9 if ignore_terminate else -15)
        with pytest.raises(ChildProcessError):
            os.waitpid(processes[0].pid, os.WNOHANG)
    finally:
        monitor.close()


def test_cancelled_disk_scan_does_not_spawn_a_process(tmp_path, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("A cancelled quota check must not start du")

    monkeypatch.setattr(evaluation.subprocess, "Popen", unexpected_popen)
    with pytest.raises(evaluation.DiskCheckCancelled):
        evaluation.directory_usage(tmp_path, cancel_event=cancelled)


def test_two_nodes_eight_gpus_cover_all_ranks():
    ranks = []
    for node in range(2):
        topo = evaluation.topology({
            "TOTAL_GPUS": "16", "NNODES": "2", "GPUS_PER_NODE": "8",
            "NODE_RANK": str(node),
        })
        assert topo["nodes"] == 2 and topo["local"] == 8
        ranks.extend(topo["node_rank"] * topo["local"] + local for local in topo["selected"])
    assert ranks == list(range(16))


def test_externally_spawned_workers_cover_exactly_16_ranks():
    ranks = []
    for rank in range(16):
        topo = evaluation.topology({
            "TOTAL_GPUS": "16", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8",
            "RANK": str(rank), "LOCAL_RANK": str(rank % 8),
        })
        assert topo["selected"] == [rank % 8]
        ranks.append(topo["node_rank"] * topo["local"] + topo["selected"][0])
    assert ranks == list(range(16))


@pytest.mark.parametrize("env", [
    {"NNODES": "3", "GPUS_PER_NODE": "8"},
    {"NNODES": "2", "GPUS_PER_NODE": "8", "NODE_RANK": "2"},
    {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8", "RANK": "9", "LOCAL_RANK": "0"},
    {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"},
    {"TOTAL_GPUS": "0"},
])
def test_reject_inconsistent_topology(env):
    with pytest.raises(ValueError):
        evaluation.topology(env)


def test_resume_manifest_accepts_identical_and_rejects_fingerprint_change(config):
    path = evaluation.install_manifest(config)
    original = path.read_bytes()
    assert evaluation.install_manifest(copy.deepcopy(config)) == path
    changed = dict(config, identity="changed-checkpoint-or-protocol")
    with pytest.raises(ValueError, match="different configuration"):
        evaluation.install_manifest(changed)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".config.*.json"))


def test_resume_result_verifies_identity_job_and_status(config):
    job = evaluation.jobs_for(config)[0]
    assert evaluation.read_result(config["output_dir"], job, config["identity"]) is None
    write_result(config, job, "success")
    assert evaluation.read_result(config["output_dir"], job, config["identity"])["status"] == "success"
    with pytest.raises(ValueError, match="Mismatched"):
        evaluation.read_result(config["output_dir"], job, "other-fingerprint")
    write_result(config, job, "success", job_id="different-episode")
    with pytest.raises(ValueError, match="Mismatched"):
        evaluation.read_result(config["output_dir"], job, config["identity"])
    write_result(config, job, "running")
    with pytest.raises(ValueError, match="Invalid terminal status"):
        evaluation.read_result(config["output_dir"], job, config["identity"])


def test_errors_and_pending_remain_in_planned_success_denominator(config):
    jobs = evaluation.jobs_for(config)
    for job, status in zip(jobs, ("success", "timeout", "error")):
        write_result(config, job, status)
    summary = evaluation.summarize(config, write=False)
    all_tasks = summary["groups"]["all"]
    assert all_tasks["planned"] == 1500
    assert all_tasks["success"] == all_tasks["timeout"] == all_tasks["error"] == 1
    assert all_tasks["pending"] == 1497
    assert all_tasks["success_rate_planned"] == pytest.approx(1 / 1500)
    assert all_tasks["success_rate_valid"] == pytest.approx(1 / 2)
    assert not summary["complete"]
    assert summary["groups"]["seen"]["planned"] == 1200
    assert summary["groups"]["unseen"]["planned"] == 300
    assert sum(row["planned"] for row in summary["tasks"]) == 1500


def test_complete_attempts_keep_error_count_and_planned_denominator(config):
    config["tasks"] = [config["tasks"][0]]
    config["episodes_per_task"] = 3
    for job, status in zip(evaluation.jobs_for(config), ("success", "timeout", "error")):
        write_result(config, job, status)
    summary = evaluation.summarize(config)
    assert summary["complete"]
    stats = summary["groups"]["all"]
    assert stats["error"] == 1 and stats["pending"] == 0
    assert stats["success_rate_planned"] == pytest.approx(1 / 3)
    assert stats["success_rate_valid"] == pytest.approx(1 / 2)
    output = Path(config["output_dir"])
    assert json.loads((output / "summary.json").read_text())["groups"]["all"] == stats
    assert len((output / "episodes.csv").read_text().splitlines()) == 4
