"""Shared scheduling, provenance, resume, and scoring for Unitree G1 VLA evaluation.

No policy framework or simulator is imported when planning or reading reports.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def owned(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Writable path must stay under {root}: {path}")
    return resolved

def integer(env, key, default, minimum=1):
    value = int(env.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value

def available_cpu_cores(cgroup_root=Path("/sys/fs/cgroup")):
    """Account for container CPU quota as well as scheduler affinity."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    limits = [affinity]
    root = Path(cgroup_root)
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max" and int(quota) > 0 and int(period) > 0:
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    for folder in ("cpu", "cpu,cpuacct"):
        try:
            quota = int((root / folder / "cpu.cfs_quota_us").read_text())
            period = int((root / folder / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    return min(limits)

def topology(env):
    """DLC normally launches one command per node; also support pre-spawned workers."""
    total = integer(env, "TOTAL_GPUS", 16)
    if "LOCAL_RANK" in env:
        local = integer(env, "LOCAL_WORLD_SIZE", 0)
        rank = integer(env, "RANK", -1, 0)
        local_rank = integer(env, "LOCAL_RANK", -1, 0)
        if (integer(env, "WORLD_SIZE", 0) != total or rank % local != local_rank
                or rank >= total or local_rank >= local):
            raise ValueError("Inconsistent externally launched GPU ranks")
        nodes, node_rank = total // local, rank // local
        selected = [local_rank]
    else:
        nodes = integer(env, "NNODES", env.get("WORLD_SIZE", 1))
        local = integer(env, "GPUS_PER_NODE", total // nodes)
        node_rank = integer(env, "NODE_RANK", env.get("RANK", 0), 0)
        selected = list(range(local))
    if total != nodes * local or node_rank >= nodes:
        raise ValueError("TOTAL_GPUS must equal NNODES * GPUS_PER_NODE; invalid node rank")
    return dict(total=total, nodes=nodes, node_rank=node_rank, local=local, selected=selected)

def jobs_for(config):
    """Longest horizons first, spread each task's episodes across all GPU ranks."""
    tasks = sorted(config["tasks"], key=lambda t: (t["category"] != "composite", t["name"]))
    jobs = []
    for ep in range(config["episodes_per_task"]):
        for task in tasks:
            key = f"{task['name']}__ep{ep:03d}"
            seed = int.from_bytes(hashlib.sha256(
                f"{config['seed']}:{task['name']}:{ep}".encode()).digest()[:4], "big")
            seconds = config["atomic_seconds"] if task["category"] == "atomic" else config["composite_seconds"]
            jobs.append(dict(id=key, task=task, episode=ep, seed=seed,
                             timeout_seconds=seconds,
                             max_model_steps=int(round(seconds * config["model_hz"]))))
    # Greedy static allocation balances timeout budgets; jobs within a rank use a dynamic queue.
    loads = [0] * config["total_gpus"]
    for job in sorted(jobs, key=lambda j: (-j["timeout_seconds"], j["episode"], j["task"]["name"])):
        rank = min(range(len(loads)), key=lambda r: (loads[r], r))
        job["rank"] = rank
        loads[rank] += job["timeout_seconds"]
    return jobs

def install_manifest(config):
    """Exclusive creation prevents simultaneous nodes from silently mixing configurations."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.json"
    payload = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    candidate = output / f".config.{os.getpid()}.{time.time_ns()}.json"
    candidate.write_text(payload)
    try:
        try:
            os.link(candidate, path)
        except FileExistsError:
            old = json.loads(path.read_text())
            if old.get("identity") != config["identity"]:
                raise ValueError("Existing RUN_NAME has a different configuration/checkpoint. Use a new RUN_NAME.")
    finally:
        candidate.unlink(missing_ok=True)
    return path

def result_path(output, job):
    return Path(output) / "episodes" / (job["id"] + ".json")

def read_result(output, job, identity):
    path = result_path(output, job)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if row.get("identity") != identity or row.get("job_id") != job["id"]:
        raise ValueError(f"Mismatched prior episode result: {path}")
    if row.get("status") not in {"success", "timeout", "error"}:
        raise ValueError(f"Invalid terminal status: {path}")
    return row

def summarize(config, write=True):
    if not write:
        return _summarize_unlocked(config, write=False)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Lock before reading episode records. Locking only the replacements would
    # allow a delayed older snapshot to overwrite a newer node's complete report.
    with (output / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize_unlocked(config, write=True)

def _summarize_unlocked(config, write):
    import csv
    output = Path(config["output_dir"])
    rows = []
    for job in jobs_for(config):
        record = read_result(output, job, config["identity"])
        rows.append(dict(task=job["task"]["name"], category=job["task"]["category"],
                         split=job["task"]["split"], episode=job["episode"], seed=job["seed"],
                         status=record["status"] if record else "pending",
                         success=bool(record and record["status"] == "success"),
                         sim_seconds=record.get("sim_seconds") if record else None,
                         wall_seconds=record.get("wall_seconds") if record else None,
                         reason=record.get("reason", "") if record else ""))

    def stats(items):
        n = len(items)
        success = sum(x["status"] == "success" for x in items)
        errors = sum(x["status"] == "error" for x in items)
        pending = sum(x["status"] == "pending" for x in items)
        valid = n - errors - pending
        return dict(planned=n, success=success, timeout=valid-success, error=errors,
                    pending=pending, success_rate_planned=success/n if n else None,
                    success_rate_valid=success/valid if valid else None)

    groups = {"all": stats(rows),
              "seen": stats([r for r in rows if r["split"] == "seen"]),
              "unseen": stats([r for r in rows if r["split"] == "unseen"]),
              "atomic": stats([r for r in rows if r["category"] == "atomic"]),
              "composite": stats([r for r in rows if r["category"] == "composite"]),
              "seen_composite": stats([r for r in rows if r["category"] == "composite" and r["split"] == "seen"])}
    tasks = [dict(task=t["name"], category=t["category"], split=t["split"],
                  **stats([r for r in rows if r["task"] == t["name"]])) for t in config["tasks"]]
    report = dict(identity=config["identity"], updated_at=datetime.now(timezone.utc).isoformat(),
                  complete=groups["all"]["pending"] == 0,
                  scorable_complete=groups["all"]["pending"] == groups["all"]["error"] == 0,
                  groups=groups, tasks=tasks)
    if write:
        atomic_json(output / "summary.json", report)
        for name, data in [("episodes.csv", rows), ("tasks.csv", tasks)]:
            target = output / name
            tmp = output / f".{name}.{os.getpid()}.{uuid4().hex}.tmp"
            with tmp.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            os.replace(tmp, target)
    return report

class DiskCheckCancelled(Exception):
    """An in-progress quota scan was deliberately stopped during worker cleanup."""

def directory_usage(root, cancel_event=None):
    root = Path(root).resolve()
    if cancel_event is not None and cancel_event.is_set():
        raise DiskCheckCancelled()
    command = ["du", "-s", "-B1", str(root)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=dict(os.environ, LC_ALL="C"))
    try:
        while True:
            # Preserve the exit status of a scan that already finished, even when
            # shutdown was requested before its result could be consumed.
            if (cancel_event is not None and cancel_event.is_set()
                    and process.poll() is None):
                raise DiskCheckCancelled()
            try:
                stdout, stderr = process.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    if result.returncode:
        # Cache builders remove temporary files while du is walking them.
        # Those vanished files consume no space; keep all other failures fatal.
        lines = result.stderr.splitlines()
        transient = bool(lines)
        for line in lines:
            match = re.fullmatch(r"du: cannot access '(.+)': No such file or directory", line)
            path = Path(match.group(1)) if match else None
            transient = transient and path is not None and any(
                path.is_relative_to(root / part) for part in ("cache", "tmp"))
        if result.returncode != 1 or not transient:
            raise subprocess.CalledProcessError(result.returncode, result.args,
                                                result.stdout, result.stderr)
    size = int(result.stdout.split()[0])
    if size < 0:
        raise ValueError("Invalid directory usage")
    return size

def disk_preflight(config, cancel_event=None):
    import shutil
    root = config["workspace"]
    # Conservative whole-directory allocated size; does not delete any existing content.
    used = directory_usage(root, cancel_event=cancel_event)
    free = shutil.disk_usage(root).free
    if used >= config["disk_limit_bytes"]:
        raise ValueError(f"Workspace exceeds configured quota ({used} >= {config['disk_limit_bytes']} bytes)")
    if free < config["minimum_free_bytes"]:
        raise ValueError(f"Shared disk has insufficient free space: {free} bytes")
    return dict(user_directory_bytes=used, filesystem_free_bytes=free,
                directory_limit_bytes=config["disk_limit_bytes"])


def load_tasks(path=None):
    from .configuration import load_tasks as load
    return load(path)


def prepare_workspace(config):
    """Keep caches, temporary files, model logs, and generated data in one workspace."""
    root = Path(config["workspace"])
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "weights/hub" if (root / "weights/hub").is_dir() else root / "cache/huggingface/hub"
    locations = {
        "TMPDIR": root / "tmp", "TMP": root / "tmp", "TEMP": root / "tmp",
        "HF_HOME": root / "cache/huggingface", "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub, "HF_ASSETS_CACHE": root / "cache/huggingface/assets",
        "HF_DATASETS_CACHE": root / "cache/huggingface/datasets",
        "XDG_CACHE_HOME": root / "cache/xdg", "TORCH_HOME": root / "cache/torch",
        "TRITON_CACHE_DIR": root / "cache/triton", "NUMBA_CACHE_DIR": root / "cache/numba-eval",
        "MPLCONFIGDIR": root / "cache/matplotlib", "MESA_SHADER_CACHE_DIR": root / "cache/mesa",
    }
    for name, path in locations.items():
        path = owned(path, root)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                      TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.dont_write_bytecode = True


def sim_environment(config, rank=0):
    env = dict(os.environ)
    root = Path(config["workspace"])
    runtime = Path(config["sim_root"]) if config.get("sim_root") else None
    paths = []
    if runtime is not None:
        paths.extend(str(runtime / name) for name in ("robocasa", "robosuite", "robosuite_models", "sonic"))
    paths.append(str(PACKAGE_ROOT))
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               MUJOCO_GL=config["renderer"], PYOPENGL_PLATFORM=config["renderer"],
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", LP_NUM_THREADS="1",
               MESA_SHADER_CACHE_DIR=str(root / "cache/mesa"),
               NUMBA_CACHE_DIR=str(root / "cache/numba-eval" / f"rank-{rank}"),
               TMPDIR=str(root / "tmp"), TMP=str(root / "tmp"), TEMP=str(root / "tmp"),
               XDG_CACHE_HOME=str(root / "cache/xdg"))
    if config["renderer"] == "osmesa":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        # PPU/CUDA shims must not enter the independent CPU simulator process.
        mesa = runtime / "mesa/usr/lib/x86_64-linux-gnu" if runtime else None
        env["LD_LIBRARY_PATH"] = str(mesa) if mesa and mesa.is_dir() else ""
    else:
        # EGL needs accelerator visibility. Its local device ordinal follows the worker rank.
        env["MUJOCO_EGL_DEVICE_ID"] = str(rank % config.get("gpus_per_node", config["total_gpus"]))
    if runtime and (runtime / "cyclonedds/install").is_dir():
        env["CYCLONEDDS_HOME"] = str(runtime / "cyclonedds/install")
    return env


def preflight(config, topo, require_gpu=False):
    """Probe the configured simulator; accelerator/framework checks belong to the policy."""
    prepare_workspace(config)
    quota = disk_preflight(config)
    for key in ("sim_python", "decoder_path"):
        if not config.get(key) or not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: configure the simulator environment and SONIC decoder")
    from .policy import resolve_factory
    resolve_factory(config["policy"]["factory"])
    work = owned(Path(config["workspace"]) / "tmp/preflight", config["workspace"])
    work.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([config["sim_python"], "-B", "-m", "robocasa_eval.sim_worker", "--check"],
                            input=json.dumps(config), capture_output=True, text=True, timeout=120,
                            env=sim_environment(config), cwd=work)
    if result.returncode:
        raise ValueError(f"Simulator preflight failed:\n{result.stderr[-8000:]}\n{result.stdout[-3000:]}")
    # Native libraries may print startup banners before the final JSON line.
    info = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "runtime_identity" in candidate:
            info = candidate
            break
    if info is None:
        raise ValueError("Simulator preflight did not provide a runtime identity")
    config["runtime_identity"] = info["runtime_identity"]
    from .configuration import refresh_identity
    refresh_identity(config)
    return dict(simulator=info, disk=quota,
                cpu=dict(effective_cores=available_cpu_cores(),
                         simulator_processes=topo["local"] * config["envs_per_gpu"],
                         reserved_model_cores_per_gpu=config["cpu_reserved_per_gpu"]),
                device_family=config["device_prefix"],
                policy_note="Device and model validation occur in the selected adapter during smoke/run")


def smoke_receipt_path(config):
    from .configuration import gate_protocol
    return Path(config["workspace"]) / "evaluation/smoke" / (digest(gate_protocol(config)) + ".json")


def require_smoke(config):
    from .configuration import gate_protocol
    path = smoke_receipt_path(config)
    if not path.is_file():
        raise ValueError("Run a single-device smoke --full-episodes for this model, runtime, cameras and protocol first")
    gate = json.loads(path.read_text())
    if (gate.get("status") != "passed" or gate.get("full_horizons") is not True
            or gate.get("protocol") != gate_protocol(config)
            or gate.get("tested_environment_count", 0) < 2):
        raise ValueError("No matching completed multi-environment, full-horizon smoke test")
    categories = {t["category"] for t in config["tasks"]}
    if set(gate.get("tested_categories", [])) != categories:
        raise ValueError("Smoke did not cover the configured task categories")
    return gate


def launch(config, topo):
    from contextlib import ExitStack
    checked = preflight(config, topo)
    require_smoke(config)
    manifest = install_manifest(config)
    output = Path(config["output_dir"])
    atomic_json(output / f"preflight-node-{topo['node_rank']}.json", checked)
    children = []
    interrupted = False
    stop_deadline = None

    def signal_group(child, signum):
        # Each worker starts its own session; its simulators belong to this group.
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    def stop(signum, frame):
        nonlocal interrupted, stop_deadline
        interrupted = True
        if stop_deadline is None:
            stop_deadline = time.monotonic() + 10
        for child in children:
            signal_group(child, signal.SIGTERM)

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ExitStack() as stack:
            for gpu in topo["selected"]:
                if interrupted:
                    break
                rank = topo["node_rank"] * topo["local"] + gpu
                log_path = output / "logs" / f"gpu-{rank:02d}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                device = "cpu" if config["device_prefix"] == "cpu" else f"{config['device_prefix']}:{gpu}"
                command = [sys.executable, "-B", "-m", "robocasa_eval.gpu_worker",
                           "--config", str(manifest), "--rank", str(rank), "--device", device]
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), env.get("PYTHONPATH", "")])
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, cwd=output, start_new_session=True)
                children.append(child)
                if interrupted:
                    signal_group(child, signal.SIGTERM)
                print(f"Worker rank {rank}: {config['envs_per_gpu']} environments; log {log_path}", flush=True)
            last_summary = 0.0
            while any(c.poll() is None for c in children):
                now = time.monotonic()
                if topo["node_rank"] == 0 and now - last_summary >= 30:
                    print(json.dumps(summarize(config)["groups"]["all"]), flush=True)
                    last_summary = now
                if any(c.poll() not in (None, 0) for c in children) and not interrupted:
                    stop(signal.SIGTERM, None)
                if stop_deadline is not None and now >= stop_deadline:
                    for child in children:
                        signal_group(child, signal.SIGKILL)
                time.sleep(1)
            codes = [c.wait() for c in children]
        report = summarize(config)
        print(json.dumps(report["groups"], indent=2), flush=True)
        if interrupted or any(codes):
            raise RuntimeError(f"One or more workers stopped: {codes}; inspect logs and resume the same run")
        if report["groups"]["all"]["error"]:
            raise RuntimeError("Evaluation completed with episode errors; inspect the report before using its success rate")
        return report
    finally:
        try:
            for child in children:
                signal_group(child, signal.SIGTERM)
            cleanup_deadline = time.monotonic() + 10
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=max(0.01, cleanup_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        signal_group(child, signal.SIGKILL)
                        child.wait()
                # Also stop a simulator left behind by an abruptly exited worker.
                signal_group(child, signal.SIGKILL)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
