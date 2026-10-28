"""CPU-only launcher lifecycle checks, including owned descendant processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from robocasa_eval import evaluation


REPORT = {"groups": {"all": {"error": 0, "pending": 0, "success": 1}}}
CHILD = """
import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))
time.sleep(60)
"""
WORKER = """
import os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-B', '-c', sys.argv[1], sys.argv[2]])
ready = Path(sys.argv[2])
deadline = time.monotonic() + 5
while not ready.exists():
    if time.monotonic() > deadline:
        child.kill()
        child.wait()
        raise RuntimeError('CPU child did not initialize')
    time.sleep(.005)
if sys.argv[3] == 'fail':
    os._exit(7)
time.sleep(60)
"""


@pytest.fixture
def launcher_config(tmp_path, monkeypatch):
    config = dict(output_dir=str(tmp_path / "run"), identity="launcher-fixture",
                  device_prefix="cpu", envs_per_gpu=2)
    topo = dict(node_rank=0, local=2, selected=[0, 1])
    monkeypatch.setattr(evaluation, "preflight", lambda *a: {"probe": "mock"})
    monkeypatch.setattr(evaluation, "require_smoke", lambda *a: {"status": "passed"})
    monkeypatch.setattr(evaluation, "summarize", lambda *a: REPORT)
    return config, topo


def is_running(pid):
    try:
        # A killed orphan can briefly remain a zombie until its parent/init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("trigger", ["failed_worker", "interrupt"])
def test_launcher_cleans_real_worker_and_descendant_groups(launcher_config, monkeypatch, trigger):
    config, topo = launcher_config
    original_popen = subprocess.Popen
    processes, descendants, groups = [], [], []
    # Accelerate only the launcher's shutdown clock, not subprocess wait internals.
    tick = [1000.0]
    def clock():
        tick[0] += 20
        return tick[0]
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(
        monotonic=clock, sleep=lambda _: time.sleep(.01), time_ns=time.time_ns))

    def controlled_worker(command, **kwargs):
        assert command[2:4] == ["-m", "robocasa_eval.gpu_worker"]
        assert kwargs["start_new_session"] is True
        index = len(processes)
        marker = Path(config["output_dir"]) / f"descendant-{index}.json"
        mode = "fail" if trigger == "failed_worker" and index == 0 else "hang"
        process = original_popen([sys.executable, "-B", "-c", WORKER, CHILD, str(marker), mode], **kwargs)
        processes.append(process)
        deadline = time.monotonic() + 5
        while not marker.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Controlled worker failed to initialize")
            time.sleep(.005)
        # Atomicity is immaterial to the production code, but wait for our child
        # to finish its marker write before parsing the controlled fixture.
        while True:
            try:
                child = json.loads(marker.read_text())
                break
            except ValueError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.005)
        descendants.append(child["pid"])
        groups.append(child["pgid"])
        assert child["pgid"] == process.pid  # Simulator really inherits its worker's group.
        return process

    monkeypatch.setattr(evaluation.subprocess, "Popen", controlled_worker)
    if trigger == "interrupt":
        invoked = []
        def interrupted_summary(*args):
            if not invoked:
                invoked.append(True)
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return REPORT
        monkeypatch.setattr(evaluation, "summarize", interrupted_summary)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with pytest.raises(RuntimeError, match="workers stopped"):
            evaluation.launch(config, topo)
        assert len(processes) == len(descendants) == 2
        assert all(process.poll() is not None for process in processes)
        deadline = time.monotonic() + 5
        while any(is_running(pid) for pid in descendants):
            assert time.monotonic() < deadline, "Launcher left an active simulator descendant"
            time.sleep(.01)
        assert {sig: signal.getsignal(sig) for sig in previous} == previous
        assert all(group == process.pid for group, process in zip(groups, processes))
    finally:
        # Never leak the controlled test subprocesses if an assertion fails.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def test_launcher_spawn_failure_kills_existing_group(launcher_config, monkeypatch):
    config, topo = launcher_config
    events = []
    process = SimpleNamespace(pid=912345, status=None)
    process.poll = lambda: process.status
    def wait(timeout=None):
        events.append(("wait", timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired("worker", timeout)
        assert process.status == -signal.SIGKILL
        return process.status
    process.wait = wait
    calls = []
    def popen(command, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise OSError("second worker failed to spawn")
        return process
    def killpg(pid, signum):
        assert pid == process.pid
        events.append(("signal", signum))
        if signum == signal.SIGKILL:
            process.status = -signum
    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluation.os, "killpg", killpg)
    with pytest.raises(OSError, match="failed to spawn"):
        evaluation.launch(config, topo)
    assert events[0] == ("signal", signal.SIGTERM)
    assert events[1][0] == "wait" and 0 < events[1][1] <= 10
    assert events[2] == ("signal", signal.SIGKILL)
    assert process.status == -signal.SIGKILL
    assert calls[0]["start_new_session"] is True
    assert all(kwargs["stdout"].closed for kwargs in calls)


def test_launcher_cleans_descendant_group_after_successful_worker_exit(launcher_config, monkeypatch):
    config, topo = launcher_config
    topo = dict(topo, selected=[0])
    process = SimpleNamespace(pid=912346, poll=lambda: 0, wait=lambda: 0)
    signals = []
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(evaluation.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    assert evaluation.launch(config, topo) == REPORT
    # The group may outlive the finished worker; do not condition this on poll().
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_interrupt_during_spawn_stops_new_child_and_skips_later_workers(launcher_config, monkeypatch):
    config, topo = launcher_config
    process = SimpleNamespace(pid=912347, status=None)
    process.poll = lambda: process.status
    process.wait = lambda: process.status
    calls, signals = [], []
    def popen(command, **kwargs):
        calls.append(kwargs)
        assert len(calls) == 1, "A cancelled launch must not start additional workers"
        # Interrupt before Popen returns, while the new child is not yet tracked.
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return process
    def killpg(pid, signum):
        assert pid == process.pid
        signals.append(signum)
        process.status = -signum
    monkeypatch.setattr(evaluation.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluation.os, "killpg", killpg)
    with pytest.raises(RuntimeError, match="workers stopped"):
        evaluation.launch(config, topo)
    assert len(calls) == 1
    assert signals[0] == signal.SIGTERM  # Newly returned child receives pending stop.
    assert signals[-1] == signal.SIGKILL  # Descendants also get final group cleanup.
    assert calls[0]["stdout"].closed
