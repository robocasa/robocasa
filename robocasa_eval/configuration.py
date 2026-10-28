"""Portable, framework-free evaluation configuration and content identities."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

from .evaluation import HERE, available_cpu_cores, digest, owned, topology

DEFAULTS = {
    "workspace": "./evaluation_runs",
    "run_name": "g1-sonic50",
    "total_gpus": 16,
    "envs_per_gpu": "auto",
    "episodes_per_task": 30,
    "atomic_seconds": 40,
    "composite_seconds": 180,
    "execution_horizon": 20,
    "batch_wait_ms": 10,
    "seed": 20261020,
    "cpu_reserved_per_gpu": 2,
    "operation_timeout_seconds": 600,
    "sim_python": sys.executable,
    "sim_root": None,
    "decoder_path": None,
    "renderer": "osmesa",
    "camera_names": ["ego_view"],
    "camera_width": 640,
    "camera_height": 480,
    "layout_ids": -1,
    "style_ids": None,
    "device_prefix": "cpu",
    "disk_limit_bytes": 2_000_000_000_000,
    "minimum_free_bytes": 10 * 1024**3,
    "task_manifest": None,
    "policy": {
        "factory": "http",
        "id": None,
        "kwargs": {"endpoint": "http://127.0.0.1:8000"},
        "artifacts": [],
    },
}

# Changes to these fields require a new real-environment integration test.
# Run names, rank counts, and output locations may change after single-device smoke.
GATE_KEYS = (
    "schema_version", "embodiment", "policy", "policy_identity", "code_sha256",
    "runtime_sources", "runtime_identity", "remote_policy_identity", "decoder_sha256", "sim_python",
    "sim_root", "model_hz", "control_hz", "physics_dt", "execution_horizon",
    "atomic_seconds", "composite_seconds", "renderer", "camera_names",
    "camera_width", "camera_height", "layout_ids", "style_ids", "tasks",
    "task_manifest_identity", "device_prefix",
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root, skip):
    """Include symlinked checkpoint subdirectories, rejecting recursive links."""
    def visit(folder, ancestors):
        resolved = folder.resolve()
        if resolved in ancestors:
            raise ValueError(f"Recursive directory symlink in identity input: {folder}")
        for path in sorted(folder.iterdir()):
            if path.name in skip or path.name.startswith(".pytest"):
                continue
            if path.is_dir():
                yield from visit(path, ancestors | {resolved})
            elif path.is_file():
                yield path
            else:
                raise ValueError(f"Identity input contains a non-regular file: {path}")
    yield from visit(Path(root), set())


def source_tree_identity(root):
    """Hash source/config and robot structure, excluding mesh/texture bytes."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Missing implementation directory: {root}")
    skip = {".git", "__pycache__", "cache", ".cache", ".venv",
            "datasets", "node_modules", "logs", "weights", "runs"}
    files = {}
    for p in walk_files(root, skip):
        if p.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".xml", ".urdf", ".srdf"}:
            files[str(p.relative_to(root))] = sha256_file(p)
    return {"path": str(root), "source_sha256": digest(files), "source_files": len(files)}


def artifact_identity(path):
    """Pin model weights and processors by their bytes, regardless of model format."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Missing policy artifact: {root}")
    if root.is_file():
        return {"path": str(root), "bytes": root.stat().st_size, "sha256": sha256_file(root)}
    files = {}
    for p in walk_files(root, {".git", "__pycache__", ".cache"}):
        files[str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    if not files:
        raise ValueError(f"Empty policy artifact directory: {root}")
    return {"path": str(root), "files": files, "sha256": digest(files)}


def load_tasks(path=None):
    data = json.loads(Path(path or HERE / "tasks50.json").read_text())
    if not isinstance(data, dict):
        raise ValueError("Task manifest must be a JSON object")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest must contain a nonempty tasks list")
    names = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"name", "category", "split"}:
            raise ValueError("Each task requires exactly name, category, split")
        if not isinstance(task["name"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task["name"]):
            raise ValueError("Invalid task class name")
        if task["name"] in names:
            raise ValueError(f"Duplicate task: {task['name']}")
        names.add(task["name"])
        if task["category"] not in ("atomic", "composite") or task["split"] not in ("seen", "unseen"):
            raise ValueError("Task category/split must be atomic|composite and seen|unseen")
    if path is None:
        counts = Counter((t["category"], t["split"]) for t in tasks)
        if len(tasks) != 50 or counts != {("atomic", "seen"): 25,
                ("composite", "seen"): 15, ("composite", "unseen"): 10}:
            raise ValueError("The packaged G1 SONIC 50-task benchmark was modified")
    return data


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def policy_identity(policy, verify_files, workspace=None):
    factory = policy["factory"]
    aliases = {"http": "robocasa_eval.adapters.http:create_policy"}
    target = aliases.get(factory, factory)
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", target):
        raise ValueError("policy.factory must be http or module:function")
    result = {"id": policy["id"], "factory": target, "verified": verify_files}
    if not verify_files:
        return result
    # Deliberately not executed for plan: external policy packages may import a framework.
    module, _ = target.split(":")
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        raise ValueError(f"Cannot locate policy factory module: {module}")
    result["adapter_source"] = {"path": spec.origin, "sha256": sha256_file(spec.origin)}
    result["artifacts"] = [artifact_identity(p) for p in sorted(set(policy["artifacts"]))]
    return result


def normalize_policy(policy):
    """Default to the HTTP service; keep custom factory options opaque."""
    policy = copy.deepcopy(policy)
    if not isinstance(policy, dict) or set(policy) - {"factory", "id", "kwargs", "artifacts"}:
        raise ValueError("policy accepts factory, id, kwargs, artifacts")
    policy = dict(policy)
    policy.setdefault("factory", "http")
    policy.setdefault("id", None)
    policy.setdefault("kwargs", {})
    policy.setdefault("artifacts", [])
    if not isinstance(policy["factory"], str) or not policy["factory"].strip():
        raise ValueError("policy.factory must be http or module:function")
    if policy["id"] is not None and (not isinstance(policy["id"], str) or not policy["id"].strip()):
        raise ValueError("policy.id must be null or a nonempty revision string")
    if (not isinstance(policy["kwargs"], dict)
            or any(not isinstance(key, str) for key in policy["kwargs"])
            or set(policy["kwargs"]) & {"device", "seed", "workspace"}):
        raise ValueError("policy.kwargs must use string keys without device/seed/workspace overrides")
    if not isinstance(policy["artifacts"], list) or any(not isinstance(p, str) for p in policy["artifacts"]):
        raise ValueError("policy.artifacts must be a list of local file/directory paths")
    # Paths are relative to the invoking working directory, consistently with the CLI.
    policy["artifacts"] = [str(Path(p).expanduser().resolve()) for p in policy["artifacts"]]
    if policy["factory"] in {"http", "robocasa_eval.adapters.http:create_policy"}:
        policy["kwargs"].setdefault("endpoint", "http://127.0.0.1:8000")
    return policy


def build_config(options=None, *, verify_files=False, environment=None):
    """Build a plan without importing a model, NumPy, MuJoCo, or PyTorch."""
    if options is not None and not isinstance(options, dict):
        raise ValueError("Evaluation options must be a mapping")
    options = copy.deepcopy(options or {})
    for alias, canonical in (("num_workers", "total_gpus"), ("envs_per_worker", "envs_per_gpu")):
        if alias in options:
            if canonical in options and options[canonical] != options[alias]:
                raise ValueError(f"Conflicting {alias} and {canonical}")
            options[canonical] = options.pop(alias)
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown evaluation options: {sorted(unknown)}")
    raw = json.loads(json.dumps(DEFAULTS))
    raw.update(options)
    root = Path(raw["workspace"]).expanduser().resolve()
    run = raw["run_name"]
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run):
        raise ValueError("run_name must be a simple directory name")
    for key in ("total_gpus", "episodes_per_task", "execution_horizon", "operation_timeout_seconds",
                "camera_width", "camera_height", "disk_limit_bytes"):
        _positive_int(raw[key], key)
    for key in ("seed", "batch_wait_ms", "cpu_reserved_per_gpu", "minimum_free_bytes"):
        _positive_int(raw[key], key, 0)
    if raw["camera_width"] * raw["camera_height"] > 4096 * 4096:
        raise ValueError("Camera dimensions exceed the supported IPC image budget")
    for key in ("atomic_seconds", "composite_seconds"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        if not math.isclose(value * 50, round(value * 50), abs_tol=1e-8):
            raise ValueError(f"{key} must be an exact multiple of 1/50 second")
    if raw["renderer"] not in ("osmesa", "egl"):
        raise ValueError("renderer must be osmesa or egl")
    cameras = raw["camera_names"]
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(c, str) or c not in {"ego_view", "left_wrist", "wrist_view"} for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("camera_names must select unique supported G1 cameras")
    if raw["camera_width"] * raw["camera_height"] * 3 * len(cameras) > 24 * 1024**2:
        raise ValueError("Camera images exceed the 32 MiB simulator IPC frame budget")
    for key in ("layout_ids", "style_ids"):
        if (raw[key] is None and key == "layout_ids") or (raw[key] is not None and
                (isinstance(raw[key], bool) or not isinstance(raw[key], int))):
            raise ValueError(f"{key} must be an integer (style_ids may be null)")
    if not isinstance(raw["device_prefix"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", raw["device_prefix"]):
        raise ValueError("device_prefix must be a device family such as cuda, cpu, or xpu")
    env = dict(os.environ if environment is None else environment)
    env["TOTAL_GPUS"] = str(raw["total_gpus"])
    topo = topology(env)
    if raw["envs_per_gpu"] == "auto":
        count = min(8, max(1, available_cpu_cores() // topo["local"] - raw["cpu_reserved_per_gpu"]))
    else:
        count = _positive_int(raw["envs_per_gpu"], "envs_per_gpu")
        if count > 32:
            raise ValueError("envs_per_gpu must be <=32")
    policy = normalize_policy(raw["policy"])
    manifest = load_tasks(raw["task_manifest"])
    config = dict(raw, schema_version=2, embodiment="unitree_g1_sonic",
                  workspace=str(root), output_dir=str(owned(root / "eval_results" / run, root)),
                  policy=policy, envs_per_gpu=count, gpus_per_node=topo["local"], tasks=manifest["tasks"],
                  task_manifest_identity={k: v for k, v in manifest.items() if k != "tasks"},
                  benchmark="g1_sonic50" if raw["task_manifest"] is None else "custom_tasks",
                  model_hz=50, control_hz=200, physics_dt=0.005,
                  video_recording=False,
                  metric_definition="First native task reward > 0 within simulation-time horizon; all planned episodes remain in denominator",
                  rng_note="Scene seeds are stable per episode. Model stochasticity may depend on rank and dynamic batch ordering.")
    # A virtualenv interpreter often symlinks to the system binary. Resolving
    # that link loses pyvenv.cfg and launches Python without the sim packages.
    config["sim_python"] = str(Path(raw["sim_python"]).expanduser().absolute()) if raw["sim_python"] else None
    for key in ("sim_root", "decoder_path", "task_manifest"):
        config[key] = str(Path(raw[key]).expanduser().resolve()) if raw[key] else None
    config["policy_identity"] = policy_identity(policy, verify_files, workspace=root)
    config["code_sha256"] = {str(p.relative_to(HERE)): sha256_file(p) for p in sorted(HERE.rglob("*.py"))}
    manifest_path = Path(config["sim_root"]) / "source_manifest.json" if config["sim_root"] else None
    config["runtime_sources"] = json.loads(manifest_path.read_text()) if manifest_path and manifest_path.is_file() else None
    config["runtime_identity"] = None  # populated by a simulator-side import/source probe
    config["remote_policy_identity"] = None  # populated from the model service at preflight
    config["decoder_sha256"] = None
    if verify_files:
        for key in ("sim_python", "decoder_path"):
            if not config[key] or not Path(config[key]).is_file():
                raise ValueError(f"Missing {key}; configure the simulator Python and SONIC decoder")
        config["decoder_sha256"] = sha256_file(config["decoder_path"])
    config["identity"] = digest(config)
    return config, topo


def gate_protocol(config):
    return {key: config[key] for key in GATE_KEYS}


def refresh_identity(config):
    config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
    return config
