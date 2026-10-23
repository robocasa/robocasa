#!/usr/bin/env python3
"""One model per GPU, asynchronous CPU simulators, dynamic inference microbatches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from .evaluation import (DiskCheckCancelled, atomic_json, disk_preflight,
                        jobs_for, read_result, result_path, sim_environment)
from .sim_worker import receive, send
from .policy import EpisodeContext, load_policy, validate_action_batch


class DiskQuotaMonitor:
    """Run at most one cancellable scan without occupying simulator RPC threads."""

    def __init__(self, config, interval=600, *, check=None, clock=None):
        self.config = config
        self.interval = interval
        self.check = check or disk_preflight
        self.clock = clock or time.monotonic
        self.last_check = self.clock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="disk-quota")
        self.cancel_event = threading.Event()
        self.future = None
        self.closed = False

    def poll(self):
        if self.closed:
            raise RuntimeError("Quota monitor is closed")
        if self.future is not None:
            if not self.future.done():
                return
            future, self.future = self.future, None
            future.result()  # Quota and filesystem failures must stop the worker.
            self.last_check = self.clock()
        if self.clock() - self.last_check >= self.interval:
            self.future = self.executor.submit(
                self.check, self.config, cancel_event=self.cancel_event)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel_event.set()
        future, self.future = self.future, None
        try:
            if future is not None:
                try:
                    future.result()
                except DiskCheckCancelled:
                    pass
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


class Simulator:
    def __init__(self, config, rank, slot, executor):
        self.config, self.rank, self.slot = config, rank, slot
        self.executor = executor
        self.job = None
        self.context = None
        self.observation = None
        self.future = None
        self.request_kind = None
        self.start_wall = self.request_wall = 0.0
        self.last_steps = 0
        work = Path(config["output_dir"]) / "workers" / f"gpu-{rank:02d}" / f"env-{slot:02d}"
        work.mkdir(parents=True, exist_ok=True)
        self.log = (work / "simulator.log").open("a")
        self.proc = None
        try:
            env = sim_environment(config, rank)
            env["NUMBA_CACHE_DIR"] = str(Path(config["workspace"]) / "cache/numba-eval" / f"gpu-{rank}-env-{slot}")
            Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [config["sim_python"], "-B", "-m", "robocasa_eval.sim_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, env=env, cwd=work, bufsize=0)
            self.call({"command": "init", "config": dict(config, work_dir=str(work))})
            result = self.future.result(timeout=config["operation_timeout_seconds"])
            self.future = None
            if not result.get("ok"):
                raise RuntimeError(f"Simulator initialization failed: {result}")
        except BaseException:
            self.close()
            raise

    def call(self, request):
        if self.future is not None:
            raise RuntimeError("Only one outstanding operation per simulator")
        self.request_wall = time.monotonic()
        self.request_kind = request["command"]

        def rpc():
            send(self.proc.stdin, request)
            return receive(self.proc.stdout)

        self.future = self.executor.submit(rpc)

    def reset(self, job):
        self.job, self.observation = job, None
        self.start_wall = time.monotonic()
        self.last_steps = 0
        self.call({"command": "reset", "job": job})

    def close(self):
        """Stop the owned child before joining its outstanding pipe operation."""
        proc = self.proc
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            future, self.future = self.future, None
            if future is not None:
                future.cancel()
                try:
                    future.result(timeout=5)
                except Exception:
                    pass  # Terminating a child normally closes its pending RPC.
            streams = (proc.stdin, proc.stdout, self.log) if proc else (self.log,)
            for stream in streams:
                if stream is not None:
                    stream.close()


def run(config, rank, device):
    """Own one policy and several independent episode contexts for this rank."""
    output = Path(config["output_dir"])
    lock_path = output / f"gpu-{rank:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Include the no-work and policy-load paths in lock lifetime management.
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(config, rank, device, output)


def _run_locked(config, rank, device, output):
    jobs = [j for j in jobs_for(config) if j["rank"] == rank and
            read_result(output, j, config["identity"]) is None]
    if not jobs:
        print(f"rank {rank}: all assigned episodes already complete", flush=True)
        return
    policy = executor = disk_monitor = None
    slots = []
    stopping = False
    total_batches = total_samples = 0
    model_seconds = 0.0
    last_status = 0.0
    queue = iter(jobs)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_signals = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def end_context(slot):
        # Clear first: even a failing adapter cleanup is called only once.
        context, slot.context = slot.context, None
        if context is not None:
            policy.end_episode(context)

    def finish(slot, status, response):
        if slot.job is None:
            return
        response = dict(response)
        try:
            end_context(slot)
        except Exception as exc:
            status = "error"
            response["reason"] = f"policy.end_episode failed: {exc}"
            response["traceback"] = traceback.format_exc()
        record = dict(identity=config["identity"], job_id=slot.job["id"], rank=rank,
                      slot=slot.slot, task=slot.job["task"]["name"], episode=slot.job["episode"],
                      seed=slot.job["seed"], status=status,
                      timeout_seconds=slot.job["timeout_seconds"],
                      wall_seconds=time.monotonic()-slot.start_wall,
                      sim_seconds=response.get("sim_seconds", slot.last_steps/config["model_hz"]),
                      sim_steps=response.get("sim_steps", slot.last_steps),
                      reason=response.get("reason", response.get("error", status)),
                      metadata=response.get("metadata", {}),
                      error_traceback=response.get("traceback"),
                      finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(result_path(output, slot.job), record)
        print(f"{slot.job['id']}: {status}; sim={record['sim_seconds']:.2f}s; wall={record['wall_seconds']:.1f}s", flush=True)
        slot.job = None
        slot.observation = None

    def next_job(slot):
        while not stopping:
            job = next(queue, None)
            if job is None:
                slot.job = slot.observation = None
                return
            slot.job, slot.observation = job, None
            slot.start_wall, slot.last_steps = time.monotonic(), 0
            slot.context = EpisodeContext(
                # The run fingerprint includes its output/run namespace. Shared
                # policy services must not merge histories from separate runs
                # that evaluate the same task/episode. On-disk job IDs stay local.
                episode_id=f"{config['identity']}:{job['id']}",
                task=job["task"]["name"], episode=job["episode"],
                seed=job["seed"], rank=rank, slot=slot.slot)
            try:
                policy.reset(slot.context)
            except Exception as exc:
                finish(slot, "error", dict(error=f"policy.reset failed: {exc}",
                                            traceback=traceback.format_exc()))
                continue
            if stopping:
                return
            slot.reset(job)
            return

    def write_status():
        atomic_json(output / "status" / f"gpu-{rank:02d}.json", dict(
            identity=config["identity"], rank=rank, active_episodes=[s.job["id"] for s in slots if s.job],
            inference_batches=total_batches, inference_samples=total_samples,
            mean_batch_size=total_samples/total_batches if total_batches else 0,
            model_seconds=model_seconds, updated_at=datetime.now(timezone.utc).isoformat()))

    try:
        policy = load_policy(config, device=device, seed=config["seed"] + rank)
        executor = ThreadPoolExecutor(max_workers=config["envs_per_gpu"])
        for i in range(min(config["envs_per_gpu"], len(jobs))):
            if stopping:
                break
            slot = Simulator(config, rank, i, executor)
            slots.append(slot)
            next_job(slot)
        if rank == 0:
            disk_monitor = DiskQuotaMonitor(config)
        ready_since = None
        while any(s.job is not None for s in slots) and not stopping:
            if disk_monitor is not None:
                disk_monitor.poll()
            now = time.monotonic()
            for i, slot in enumerate(list(slots)):
                if slot.future is None:
                    continue
                expired = (not slot.future.done()
                           and now-slot.request_wall > config["operation_timeout_seconds"])
                if not slot.future.done() and not expired:
                    continue
                kind = slot.request_kind
                try:
                    if expired:
                        raise TimeoutError(f"Simulator {kind} exceeded wall watchdog (not task simulation timeout)")
                    response = slot.future.result()
                    slot.future = None
                except Exception as exc:
                    response = dict(ok=False, error=str(exc), traceback=traceback.format_exc())
                if not response.get("ok"):
                    finish(slot, "error", response)
                    slot.close()
                    if not stopping:
                        slot = Simulator(config, rank, i, executor)
                        slots[i] = slot
                        next_job(slot)
                elif kind == "reset":
                    slot.observation = response["observation"]
                elif response.get("finished"):
                    slot.last_steps = response.get("sim_steps", slot.last_steps)
                    finish(slot, "success" if response["success"] else "timeout", response)
                    next_job(slot)
                else:
                    slot.last_steps = response["sim_steps"]
                    slot.observation = response["observation"]
            ready = [s for s in slots if s.job is not None and s.observation is not None and s.future is None]
            active = [s for s in slots if s.job is not None]
            if ready and not stopping:
                if ready_since is None:
                    ready_since = time.monotonic()
                if len(ready) == len(active) or (time.monotonic()-ready_since)*1000 >= config["batch_wait_ms"]:
                    started = time.monotonic()
                    try:
                        actions = policy.predict_batch(
                            [s.observation for s in ready], [s.context for s in ready])
                        actions = validate_action_batch(
                            actions, batch_size=len(ready),
                            execution_horizon=config["execution_horizon"],
                            remaining_steps=[s.job["max_model_steps"]-s.last_steps for s in ready])
                    except Exception as exc:
                        # No silent re-prediction on a partially advanced policy history.
                        failure = dict(error=f"Policy inference/action contract failed: {exc}",
                                       traceback=traceback.format_exc())
                        for slot in ready:
                            finish(slot, "error", failure)
                        for slot in ready:
                            next_job(slot)
                    else:
                        total_batches += 1
                        total_samples += len(ready)
                        if not stopping:
                            for slot, chunk in zip(ready, actions):
                                slot.observation = None
                                slot.call({"command": "step", "actions": chunk})
                    finally:
                        model_seconds += time.monotonic()-started
                    ready_since = None
            else:
                ready_since = None
            if now-last_status > 30:
                write_status()
                last_status = now
            time.sleep(0.001)
        write_status()
        if stopping:
            raise InterruptedError("Evaluation interrupted; in-flight episodes remain pending for resume")
    except Exception as exc:
        if not stopping:
            failure = dict(error=f"Worker failed: {exc}", traceback=traceback.format_exc())
            for slot in slots:
                if slot.job is not None:
                    try:
                        finish(slot, "error", failure)
                    except Exception as write_exc:
                        print(f"Could not persist episode failure: {write_exc}", file=sys.stderr)
        raise
    finally:
        # Each resource gets a cleanup attempt even if an earlier cleanup fails.
        failures = []
        if disk_monitor is not None:
            try:
                disk_monitor.close()
            except Exception as exc:
                failures.append(exc)
        for slot in slots:
            try:
                end_context(slot)
            except Exception as exc:
                failures.append(exc)
            try:
                slot.close()
            except Exception as exc:
                failures.append(exc)
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                failures.append(exc)
        if policy is not None:
            try:
                policy.close()
            except Exception as exc:
                failures.append(exc)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        if failures:
            if sys.exc_info()[0] is None:
                raise RuntimeError(f"Worker cleanup failed: {failures}")
            print(f"Worker cleanup also failed: {failures}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.rank, args.device)


if __name__ == "__main__":
    main()
