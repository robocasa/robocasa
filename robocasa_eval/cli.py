"""Public command line for model-independent Unitree G1 evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("plan", "check", "smoke", "run", "summarize", "serve"),
                        nargs="?", default="plan")
    result.add_argument("--config", type=Path, help="JSON configuration; relative paths use the invocation directory")
    result.add_argument("--workspace")
    result.add_argument("--run-name")
    result.add_argument("--total-gpus", "--num-workers", dest="total_gpus", type=int,
                        help="Evaluation worker count; HTTP workers connect to the shared model service")
    result.add_argument("--envs-per-gpu", "--envs-per-worker", dest="envs_per_gpu",
                        help="auto or concurrent simulator count per evaluation worker")
    result.add_argument("--episodes-per-task", type=int)
    result.add_argument("--device-prefix", help="Adapter device family, e.g. cuda or cpu")
    result.add_argument("--envs", type=int, default=4, help="Smoke concurrent environments")
    result.add_argument("--steps", type=int, default=100, help="Short smoke model-step cap")
    result.add_argument("--full-episodes", action="store_true", help="Smoke with full benchmark time limits")
    result.add_argument("--tasks", nargs="+", help="Smoke task names from the configured manifest")
    result.add_argument("--host", help="Model service bind address (serve)")
    result.add_argument("--port", type=int, help="Single model service port (serve)")
    result.add_argument("--devices", nargs="+", help="Model replicas, e.g. cuda:0 cuda:1 (serve)")
    return result


def read_options(args):
    options = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(options, dict):
        raise ValueError("Configuration must be a JSON object")
    for key in ("workspace", "run_name", "total_gpus", "episodes_per_task", "device_prefix"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
            if key == "total_gpus":
                options.pop("num_workers", None)
    if args.envs_per_gpu is not None:
        options["envs_per_gpu"] = "auto" if args.envs_per_gpu == "auto" else int(args.envs_per_gpu)
        options.pop("envs_per_worker", None)
    if args.command == "serve":
        for key in ("host", "port", "devices"):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
    elif any(getattr(args, key) is not None for key in ("host", "port", "devices")):
        raise ValueError("--host/--port/--devices configure serve; evaluation uses policy.kwargs.endpoint")
    return options


def run_service(options):
    from contextlib import redirect_stderr, redirect_stdout
    import threading
    from .evaluation import atomic_json, disk_preflight, owned, prepare_workspace
    from .serving_configuration import build_server_config, service_metadata
    from .serving_http import serve
    config = build_server_config(options)
    prepare_workspace(config)
    disk_preflight(config)
    print("Checking model artifacts before opening the inference port...", file=sys.stderr, flush=True)
    config = build_server_config(options, verify_files=True)
    metadata = service_metadata(config)
    output = owned(Path(config["workspace"]) / "model_service" / metadata["fingerprint"], config["workspace"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", config)
    atomic_json(output / "metadata.json", metadata)
    # Relative outputs created by a model library also belong to the workspace.
    previous = Path.cwd()
    try:
        os.chdir(output)
        with (output / "server.log").open("a", encoding="utf-8", buffering=1) as log:
            lock = threading.RLock()
            class Tee:
                def __init__(self, stream):
                    self.stream = stream
                def write(self, text):
                    with lock:
                        log.write(text)
                        return self.stream.write(text)
                def flush(self):
                    with lock:
                        log.flush()
                        self.stream.flush()
                def isatty(self):
                    return self.stream.isatty()
                def fileno(self):
                    return self.stream.fileno()
                @property
                def encoding(self):
                    return self.stream.encoding
            with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
                serve(config, metadata)
                print("Model service stopped.", file=sys.stderr, flush=True)
    finally:
        os.chdir(previous)
    return dict(status="stopped", model_id=metadata["model_id"], output=str(output))


def main(argv=None):
    # Do not create __pycache__ alongside installed source or a model repository.
    sys.dont_write_bytecode = True
    args = parser().parse_args(argv)
    from .configuration import build_config
    from .evaluation import (atomic_json, jobs_for, launch,
                             preflight, prepare_workspace, summarize)
    try:
        options = read_options(args)
        if args.command == "serve":
            result = run_service(options)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        # A smoke command always runs on one device, even under a DLC node environment.
        env = dict(os.environ)
        if args.command in {"smoke", "summarize"}:
            for key in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE", "NODE_RANK", "NNODES", "GPUS_PER_NODE"):
                env.pop(key, None)
        # Reporting uses the saved manifest even if the model/runtime/custom task
        # files have since moved or disappeared.
        plan_options = ({k: options[k] for k in ("workspace", "run_name") if k in options}
                        if args.command == "summarize" else options)
        config, topo = build_config(plan_options, environment=env)
        if args.command == "plan":
            jobs = jobs_for(config)
            result = dict(config=config, topology=topo, total_episodes=len(jobs),
                          rank_episode_counts=dict(Counter(j["rank"] for j in jobs)),
                          rank_max_sim_seconds={r: sum(j["timeout_seconds"] for j in jobs if j["rank"] == r)
                                                for r in range(config["total_gpus"])},
                          max_concurrent_environments=config["total_gpus"] * config["envs_per_gpu"],
                          note="Plan only: no model, decoder, simulator, or hardware validation performed")
        elif args.command == "summarize":
            path = Path(config["output_dir"]) / "config.json"
            if not path.is_file():
                raise ValueError(f"No existing evaluation manifest: {path}")
            saved = json.loads(path.read_text())
            if Path(saved["output_dir"]).resolve() != path.parent.resolve():
                raise ValueError("Saved evaluation manifest points to a different output directory")
            result = summarize(saved)
        else:
            prepare_workspace(config)
            print("Checking model artifacts, adapter source, and SONIC decoder identity...", file=sys.stderr, flush=True)
            config, topo = build_config(options, verify_files=True, environment=env)
            if args.command == "check":
                result = preflight(config, topo)
                atomic_json(Path(config["workspace"]) / "evaluation/preflight.json", result)
            elif args.command == "smoke":
                from .smoke import run_smoke
                result = run_smoke(config, envs=args.envs, steps=args.steps,
                                   full_episodes=args.full_episodes, tasks=args.tasks)
            else:
                result = launch(config, topo)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
