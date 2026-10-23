"""Batch adapter for the trained GR00T N1.7 Flow / UNITREE_G1_SONIC policy.

Each process owns one policy on one device. The caller batches independent
environments; this class does not own episode history or an action queue.
GR00T returns 40-step chunks, from which the evaluator may execute a shorter
receding horizon. Hand action arrays retain the SDK's thumb-first order.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


from robocasa_eval.policy import ACTION_DIMS, STATE_DIMS, BasePolicy, EpisodeContext


LANGUAGE_KEY = "annotation.human.task_description"
ACTION_HORIZON = 40
EMBODIMENT = "unitree_g1_sonic"


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _processor_hub(root: Path) -> Path:
    configured_hub = os.environ.get("HF_HUB_CACHE")
    if configured_hub:
        candidate = Path(configured_hub).expanduser().resolve()
        if candidate.is_relative_to(root):
            return candidate
    if (root / "weights" / "hub").is_dir():
        return root / "weights" / "hub"
    return root / "cache" / "huggingface" / "hub"


def _configure_offline(root: Path) -> Path:
    """Place generated cache and temporary files in the explicit workspace."""
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    sys.dont_write_bytecode = True
    hf = root / "cache" / "huggingface"
    hub = _processor_hub(root)
    directories = {
        "HF_HOME": hf,
        "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub,
        "TRANSFORMERS_CACHE": hub,
        "HF_ASSETS_CACHE": hf / "assets",
        "HF_DATASETS_CACHE": hf / "datasets",
        "XDG_CACHE_HOME": root / "cache" / "xdg",
        "TORCH_HOME": root / "cache" / "torch",
        "TRITON_CACHE_DIR": root / "cache" / "triton",
        "TMPDIR": root / "tmp",
    }
    for name, path in directories.items():
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Workspace cache path escapes workspace: {path}")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # tempfile may have been imported before the adapter was selected.
    import tempfile
    tempfile.tempdir = str(directories["TMPDIR"])
    return hub


def _processor_snapshot(model_name: str, hub: Path) -> tuple[Path, str | None]:
    """Resolve exactly the local/offline processor snapshot used at inference."""
    model_dir = Path(model_name)
    revision = None
    if model_dir.is_dir():
        snapshot = model_dir.resolve()
    else:
        cached_repo = hub / ("models--" + model_name.replace("/", "--"))
        reference = cached_repo / "refs" / "main"
        if not reference.is_file():
            raise FileNotFoundError(
                f"Offline processor cache missing: {reference}. "
                "Prepare the Cosmos model/tokenizer cache in the dedicated "
                "workspace before launching evaluation."
            )
        revision = reference.read_text(encoding="utf-8").strip()
        snapshot = cached_repo / "snapshots" / revision
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Offline processor file missing: {snapshot / name}")
    return snapshot.resolve(), revision


def _check_processor_cache(model_name: str, hub: Path) -> Path:
    """Fail before model allocation when the VLM tokenizer cache is absent."""
    return _processor_snapshot(model_name, hub)[0]


def runtime_identity(*, checkpoint, workspace, repo=None) -> dict:
    """Read-only provenance for the actual processor cache and GR00T SDK.

    No framework or weights are loaded. Processor/tokenizer configuration,
    vocabulary, remote Python code, and chat templates are content-hashed.
    Large pretrained weight files are excluded here: the evaluated checkpoint
    is separately hashed in full by the evaluation configuration layer.
    """
    from importlib.util import find_spec
    from importlib.metadata import PackageNotFoundError, version
    from robocasa_eval.configuration import sha256_file, source_tree_identity
    from robocasa_eval.evaluation import digest

    checkpoint = Path(checkpoint).expanduser().resolve()
    workspace = Path(workspace).expanduser().resolve()
    processor_dir = checkpoint if (checkpoint / "processor_config.json").is_file() else checkpoint / "processor"
    processor = _json(processor_dir / "processor_config.json")["processor_kwargs"]
    model_name = processor["model_name"]
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("GROOT processor model_name must be a nonempty string.")
    hub = _processor_hub(workspace)
    snapshot, revision = _processor_snapshot(model_name, hub)
    files = {}
    seen_directories = set()
    weight_suffixes = {".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5", ".msgpack", ".gguf"}
    for folder, directories, names in os.walk(snapshot, followlinks=True):
        actual = Path(folder).resolve()
        if actual in seen_directories:
            raise ValueError("GROOT processor snapshot contains a directory symlink cycle or alias.")
        seen_directories.add(actual)
        directories[:] = sorted(name for name in directories if name not in {".git", "__pycache__", ".cache", ".locks"})
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix in weight_suffixes or (
                path.suffix == ".bin" and (name.startswith("pytorch_model") or name == "model.bin")
            ):
                continue
            if not path.is_file():
                raise ValueError(f"GROOT processor snapshot contains a non-file: {path}")
            files[str(path.relative_to(snapshot))] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }

    if repo is not None:
        sdk_root = Path(repo).expanduser().resolve() / "gr00t"
    else:
        # Finding a top-level package does not execute its __init__. This also
        # supports editable installations provided through import finder hooks.
        spec = find_spec("gr00t")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("Cannot locate the installed GR00T package; set policy.kwargs.repo.")
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise ValueError("Ambiguous GR00T package roots; set policy.kwargs.repo explicitly.")
        sdk_root = Path(locations[0]).resolve()
    if not (sdk_root / "policy" / "gr00t_policy.py").is_file():
        raise FileNotFoundError(f"GR00T policy source missing in {sdk_root}")
    versions = {"python": sys.version}
    for distribution in (
        "numpy", "torch", "transformers", "tokenizers", "accelerate",
        "safetensors", "huggingface_hub",
    ):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return {
        "processor": {
            "model_name": model_name, "hub": str(hub), "snapshot": str(snapshot),
            "revision": revision, "files": files, "sha256": digest(files),
        },
        "implementation": source_tree_identity(sdk_root),
        "versions": versions,
    }


def _validate_checkpoint(checkpoint: Path, inference_steps: int) -> tuple[dict, dict]:
    if not isinstance(inference_steps, int) or isinstance(inference_steps, bool) or inference_steps <= 0:
        raise ValueError("inference_steps must be a positive integer.")
    config = _json(checkpoint / "config.json")
    if config.get("model_type") != "Gr00tN1d7":
        raise ValueError("This evaluator requires a Gr00tN1d7 checkpoint.")
    if config.get("action_horizon") != ACTION_HORIZON:
        raise ValueError("Expected a 40-step SONIC action horizon.")
    if config.get("num_inference_timesteps") != inference_steps:
        raise ValueError(
            f"Checkpoint uses {config.get('num_inference_timesteps')} Flow inference "
            f"steps; requested {inference_steps}. Preserve the checkpoint setting."
        )
    for key in ("method", "objective"):
        if key in config and config[key] != "flow":
            raise ValueError(f"Expected Flow checkpoint, found {key}={config[key]!r}.")
    processor_dir = (
        checkpoint if (checkpoint / "processor_config.json").is_file()
        else checkpoint / "processor"
    )
    processor = _json(processor_dir / "processor_config.json")["processor_kwargs"]
    modalities = processor["modality_configs"].get(EMBODIMENT)
    if modalities is None:
        raise ValueError("Checkpoint lacks UNITREE_G1_SONIC processor configuration.")
    expected = {
        "video": (["ego_view"], [0]),
        "state": (list(STATE_DIMS), [0]),
        "language": ([LANGUAGE_KEY], [0]),
        "action": (list(ACTION_DIMS), list(range(ACTION_HORIZON))),
    }
    for name, (keys, times) in expected.items():
        actual = modalities[name]
        if actual["modality_keys"] != keys or actual["delta_indices"] != times:
            raise ValueError(f"Unexpected SONIC {name} modality: {actual}")
    action_configs = modalities["action"].get("action_configs") or []
    if len(action_configs) != 3 or any(a.get("rep") != "ABSOLUTE" for a in action_configs):
        raise ValueError("Expected absolute SONIC motion_token and hand actions.")
    stats = _json(processor_dir / "statistics.json")[EMBODIMENT]
    for group, dims in (("state", STATE_DIMS), ("action", ACTION_DIMS)):
        for key, dim in dims.items():
            if len(stats[group][key]["mean"]) != dim:
                raise ValueError(f"Unexpected {group}.{key} normalization dimension.")
    if _json(processor_dir / "embodiment_id.json").get(EMBODIMENT) != 11:
        raise ValueError("Expected SONIC embodiment id 11.")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        shards = set(_json(index)["weight_map"].values())
        if not shards or any(not (checkpoint / shard).is_file() for shard in shards):
            raise FileNotFoundError("Checkpoint is missing one or more model shards.")
    elif not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("Checkpoint has no safetensors model weights.")
    return config, processor


def _import_runtime(repo: str | Path | None):
    if repo is not None:
        repo_path = Path(repo).expanduser().resolve()
        if not (repo_path / "gr00t" / "policy" / "gr00t_policy.py").is_file():
            raise FileNotFoundError(f"GR00T policy source missing in {repo_path}")
        sys.path.insert(0, str(repo_path))
    torch = importlib.import_module("torch")
    module = importlib.import_module("gr00t.policy.gr00t_policy")
    if repo is not None and not Path(module.__file__).resolve().is_relative_to(repo_path):
        raise RuntimeError(
            "A different GR00T checkout is already imported. Start a fresh worker process."
        )
    return torch, module.Gr00tPolicy


class GrootPolicy(BasePolicy):
    """One stateless, shared GR00T model with native batched inference.

    Inputs use unbatched HWC uint8 RGB video, one-dimensional state arrays,
    and a plain language string. Outputs have shape (40, D) per environment.
    Flow draws noise from the process's torch RNG: seed reproducibility assumes
    the same batch sizes and ordering, which a dynamic scheduler may change.
    """

    def __init__(self, *, checkpoint, device, seed, workspace, inference_steps=4, repo=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory missing: {self.checkpoint}")
        self.inference_steps = inference_steps
        config, processor = _validate_checkpoint(self.checkpoint, inference_steps)
        self.workspace = Path(workspace).expanduser().resolve()
        hub = _configure_offline(self.workspace)
        _check_processor_cache(processor["model_name"], hub)
        self.torch, policy_class = _import_runtime(repo)
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.policy = policy_class(
            embodiment_tag="UNITREE_G1_SONIC",
            model_path=str(self.checkpoint),
            device=device,
            strict=True,
        )
        self.policy.model.eval()
        actual_steps = self.policy.model.action_head.num_inference_timesteps
        if actual_steps != inference_steps:
            raise RuntimeError(f"Loaded model uses {actual_steps} inference steps.")
        self.config = config
        self.device = device
        self.seed = seed

    def predict_batch(
        self, observations: list[dict], contexts: list[EpisodeContext]
    ) -> list[dict[str, np.ndarray]]:
        if len(observations) != len(contexts):
            raise ValueError("Observations and contexts must have the same batch length.")
        if self.policy is None:
            raise RuntimeError("GROOT policy is closed.")
        if not observations:
            return []
        images = []
        states = {key: [] for key in STATE_DIMS}
        language = []
        image_shape = None
        for index, obs in enumerate(observations):
            image = np.asarray(obs["video"]["ego_view"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Observation {index}: ego_view must be HWC uint8 RGB.")
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise ValueError(f"Observation {index}: image dimensions must be positive.")
            if image_shape is not None and image.shape != image_shape:
                raise ValueError("All batched observations must have the same image size.")
            image_shape = image.shape
            images.append(image)
            for key, dim in STATE_DIMS.items():
                value = np.asarray(obs["state"][key], dtype=np.float32)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"Observation {index}: state.{key} must be finite ({dim},).")
                states[key].append(value)
            if not isinstance(obs["language"], str) or not obs["language"].strip():
                raise ValueError(f"Observation {index}: language must be a nonempty string.")
            language.append([obs["language"]])
        batched = {
            "video": {"ego_view": np.stack(images)[:, None]},
            "state": {key: np.stack(values)[:, None] for key, values in states.items()},
            "language": {LANGUAGE_KEY: language},
        }
        with self.torch.inference_mode():
            action, _info = self.policy.get_action(batched)
        batch_size = len(observations)
        validated = {}
        for key, dim in ACTION_DIMS.items():
            value = np.asarray(action[key], dtype=np.float32)
            if value.shape != (batch_size, ACTION_HORIZON, dim):
                raise ValueError(f"Policy returned invalid action.{key} shape {value.shape}.")
            if not np.isfinite(value).all():
                raise ValueError(f"Policy returned non-finite action.{key}.")
            validated[key] = value
        # Keep independent arrays so each environment can advance its own chunk.
        return [
            {key: values[index].copy() for key, values in validated.items()}
            for index in range(batch_size)
        ]

    def close(self) -> None:
        self.policy = None


def create_policy(*, device, seed, workspace, checkpoint, inference_steps=4, repo=None):
    """Load the optional GR00T N1.7 Flow / UNITREE_G1_SONIC adapter.

    Checkpoints and the SDK checkout may be read-only paths outside workspace.
    Prepare the processor files under workspace/cache/huggingface/hub (or use an
    existing local processor path in the checkpoint). Network model downloads
    are disabled. No training-project manifests or personal paths are required.
    """
    return GrootPolicy(
        checkpoint=checkpoint, device=device, seed=seed, workspace=workspace,
        inference_steps=inference_steps, repo=repo,
    )
