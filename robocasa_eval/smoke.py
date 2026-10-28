"""Small real-environment integration test using the same worker as formal evaluation."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import traceback

from .configuration import gate_protocol, refresh_identity
from .evaluation import (atomic_json, install_manifest, owned, preflight,
                         smoke_receipt_path, summarize)


def run_smoke(config, *, envs=4, steps=100, full_episodes=False, tasks=None):
    if isinstance(envs, bool) or not isinstance(envs, int) or not 1 <= envs <= 32:
        raise ValueError("Smoke envs must be 1..32")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Smoke steps must be positive")
    by_name = {t["name"]: t for t in config["tasks"]}
    if tasks is None:
        preferred = ["CloseDishwasher", "OpenDrawer", "CoolBakedCake", "CupcakeCleanup"]
        tasks = [n for n in preferred if n in by_name]
        # Custom manifests still get every configured category in the probe.
        for category in sorted({t["category"] for t in config["tasks"]}):
            if not any(by_name[n]["category"] == category for n in tasks):
                tasks.append(next(t["name"] for t in config["tasks"] if t["category"] == category))
        if not tasks:
            tasks = list(by_name)[:4]
    if not tasks or len(set(tasks)) != len(tasks) or any(n not in by_name for n in tasks):
        raise ValueError("Smoke tasks must be distinct names from the configured task manifest")
    categories = {by_name[n]["category"] for n in tasks}
    required_categories = {t["category"] for t in config["tasks"]}
    if full_episodes and (envs < 2 or categories != required_categories):
        raise ValueError("Full smoke requires at least two concurrent environments and all configured task categories")
    formal = copy.deepcopy(config)
    formal["envs_per_gpu"] = envs
    topo = dict(total=1, nodes=1, node_rank=0, local=1, selected=[0])
    checked = preflight(formal, topo)
    protocol = gate_protocol(formal)
    receipt_path = smoke_receipt_path(formal)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config = copy.deepcopy(formal)
    config.update(run_name=f"smoke-{stamp}", total_gpus=1, gpus_per_node=1,
                  envs_per_gpu=envs, tasks=[by_name[n] for n in tasks],
                  episodes_per_task=max(1, math.ceil(envs / len(tasks))),
                  engineering_test=True, full_horizons=bool(full_episodes))
    output = owned(Path(config["workspace"]) / "eval_smoke" / config["run_name"], config["workspace"])
    output.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(output)
    if not full_episodes:
        config["atomic_seconds"] = min(config["atomic_seconds"], steps / config["model_hz"])
        config["composite_seconds"] = min(config["composite_seconds"], steps / config["model_hz"])
    refresh_identity(config)
    install_manifest(config)
    atomic_json(output / "preflight.json", checked)
    device = "cpu" if config["device_prefix"] == "cpu" else f"{config['device_prefix']}:0"
    from .gpu_worker import run
    log_path = output / "worker.log"
    print(f"Running real-model smoke on {device}; logs: {log_path}", flush=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(output)
        with log_path.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
            run(config, 0, device)
        report = summarize(config)
        if not report["scorable_complete"]:
            raise RuntimeError("Smoke contains error or pending episodes; inspect worker.log and episode records")
    except BaseException as exc:
        atomic_json(output / "failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        os.chdir(previous_directory)
    report.update(status="passed", engineering_test=True, full_horizons=bool(full_episodes),
                  output=str(output), note="Execution/integration test only, not a benchmark success estimate")
    atomic_json(output / "report.json", report)
    if full_episodes:
        atomic_json(receipt_path, dict(status="passed", full_horizons=True, engineering_test=True,
            protocol=protocol, tested_environment_count=envs, tested_categories=sorted(categories),
            tasks=list(tasks), report=str(output / "report.json"),
            timestamp=datetime.now(timezone.utc).isoformat()))
    return report
