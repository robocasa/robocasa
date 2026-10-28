"""Offline checkpoint preparation: no torch, SDK, or network is needed."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from robocasa_eval.adapters import groot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def add_weights(folder, kind):
    if kind == "safetensors":
        (folder / "model.safetensors").write_bytes(b"base model fixture")
        return ["model.safetensors"]
    if kind == "pytorch":
        (folder / "pytorch_model.bin").write_bytes(b"base model fixture")
        return ["pytorch_model.bin"]
    suffix = "safetensors" if kind == "sharded" else "bin"
    index = "model.safetensors.index.json" if kind == "sharded" else "pytorch_model.bin.index.json"
    names = [f"weights/part-{i}.{suffix}" for i in range(2)]
    for name in names:
        path = folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
    write_json(folder / index, {"weight_map": dict(zip(("layer.a", "layer.b"), names))})
    return [index, *names]


def snapshot(hub, name="nvidia/Cosmos-Reason2-2B", weights="safetensors"):
    repo = hub / ("models--" + name.replace("/", "--"))
    path = repo / "snapshots" / ("a" * 40)
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        write_json(path / filename, {})
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(path.name)
    add_weights(path, weights)
    return path


def checkpoint(root, nested=False, weights="safetensors"):
    path = root / "original-checkpoint"
    processor = path / "processor" if nested else path
    write_json(path / "config.json", {"model_name": "nvidia/Cosmos-Reason2-2B", "unchanged": [1, 2]})
    write_json(processor / "processor_config.json", {
        "processor_kwargs": {"model_name": "nvidia/Cosmos-Reason2-2B", "other": "preserve"},
    })
    write_json(processor / "statistics.json", {"normalization": [1, 2, 3]})
    write_json(processor / "embodiment_id.json", {"unitree_g1_sonic": 11})
    weight_paths = add_weights(path, weights)
    (path / "global_step1000").mkdir()
    (path / "global_step1000/optimizer.pt").write_bytes(b"unused training state")
    (path / "optimizer.pt").write_bytes(b"unused optimizer")
    (path / "trainer_state.json").write_text("{}")
    return path, processor.relative_to(path), weight_paths


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("weights", ["safetensors", "sharded"])
def test_view_rewrites_both_configs_links_inference_files_and_preserves_sources(tmp_path, nested, weights):
    original, processor, weight_paths = checkpoint(tmp_path, nested, weights)
    hub = tmp_path / "workspace/cache/huggingface/hub"
    base = snapshot(hub, weights=weights)
    original_before, base_before = hashes(original), hashes(base)
    view = groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub)

    config = json.loads((view / "config.json").read_text())
    processor_config = json.loads((view / processor / "processor_config.json").read_text())
    model_alias = Path(config["model_name"])
    processor_alias = Path(processor_config["processor_kwargs"]["model_name"])
    assert model_alias != base and model_alias.resolve() == base.resolve()
    assert processor_alias != base and processor_alias.resolve() == base.resolve()
    assert "nvidia/Cosmos-Reason2" in str(model_alias)
    assert "nvidia/Cosmos-Reason2" in str(processor_alias)
    assert model_alias.is_relative_to(view) and processor_alias.is_relative_to(view)
    assert config["unchanged"] == [1, 2]
    assert processor_config["processor_kwargs"]["other"] == "preserve"
    assert not (view / "config.json").is_symlink()
    assert not (view / processor / "processor_config.json").is_symlink()
    for name in [*weight_paths, str(processor / "statistics.json"), str(processor / "embodiment_id.json")]:
        assert (view / name).is_symlink()
        assert (view / name).resolve() == (original / name).resolve()
    assert not (view / "global_step1000").exists()
    assert not (view / "optimizer.pt").exists()
    assert not (view / "trainer_state.json").exists()
    assert hashes(original) == original_before
    assert hashes(base) == base_before
    provenance = json.loads((view / "inference_view.json").read_text())
    assert provenance["original_checkpoint"] == str(original)
    assert provenance["original_config_sha256"]["config.json"] == original_before["config.json"]
    assert provenance["base_model"]["snapshot"] == str(base)
    assert provenance["processor_model"]["revision"] == "a" * 40


@pytest.mark.parametrize("weights", ["pytorch", "pytorch_sharded"])
def test_base_model_accepts_pytorch_weights(tmp_path, weights):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    snapshot(hub, weights=weights)
    assert groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub).is_dir()


def test_model_and_processor_are_resolved_separately(tmp_path):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    base = snapshot(hub)
    other = snapshot(hub, "other/tokenizer")
    write_json(original / "processor_config.json", {"processor_kwargs": {"model_name": str(other)}})
    view = groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub)
    assert Path(json.loads((view / "config.json").read_text())["model_name"]).resolve() == base
    assert Path(json.loads((view / "processor_config.json").read_text())["processor_kwargs"]["model_name"]).resolve() == other


@pytest.mark.parametrize("name", ["nvidia/Cosmos-Reason2-2B", "Qwen/Qwen3-VL-2B-Instruct"])
def test_local_snapshot_alias_preserves_sdk_backbone_selection(tmp_path, name):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    base = snapshot(hub, name)
    # A user may already configure the snapshot path instead of a Hub ID.
    write_json(original / "config.json", {"model_name": str(base)})
    write_json(original / "processor_config.json", {"processor_kwargs": {"model_name": str(base)}})
    view = groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub)
    for file, key in (("config.json", None), ("processor_config.json", "processor_kwargs")):
        config = json.loads((view / file).read_text())
        model_name = (config[key] if key else config)["model_name"]
        # Exact dispatch condition in the real SDK's get_backbone_cls.
        assert "nvidia/Cosmos-Reason2" in model_name or "Qwen/Qwen3-VL" in model_name
        assert Path(model_name).is_dir()
        assert (Path(model_name) / "config.json").resolve() == (base / "config.json").resolve()


@pytest.mark.parametrize("invalid", [["not", "a", "path"], {"bad": "mapping"}, "../outside.safetensors", "/outside.safetensors"])
def test_bad_weight_map_values_have_clear_errors(tmp_path, invalid):
    write_json(tmp_path / "model.safetensors.index.json", {"weight_map": {"weight": invalid}})
    with pytest.raises(ValueError, match="Invalid local weight shard path"):
        groot._weight_files(tmp_path)


def test_each_load_owns_a_unique_retained_view(tmp_path):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    snapshot(hub)
    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(pool.map(lambda _: groot._prepare_inference_checkpoint(
            original, tmp_path / "workspace", hub), range(16)))
    assert len(set(views)) == 16
    assert all(path.is_dir() and (path / "inference_view.json").is_file() for path in views)


@pytest.mark.parametrize("sharded", [False, True])
def test_missing_base_weights_fail_before_runtime_import(tmp_path, monkeypatch, sharded):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    base = snapshot(hub, weights="sharded" if sharded else "safetensors")
    (base / ("weights/part-1.safetensors" if sharded else "model.safetensors")).unlink()
    monkeypatch.setattr(groot, "_validate_checkpoint", lambda *_: ({}, {}))
    monkeypatch.setattr(groot, "_configure_offline", lambda _: hub)
    monkeypatch.setattr(groot, "_import_runtime", lambda _: pytest.fail("runtime imported before cache validation"))
    with pytest.raises(FileNotFoundError, match="Offline model weights missing"):
        groot.GrootPolicy(checkpoint=original, device="cpu", seed=1, workspace=tmp_path / "workspace")
    assert not (tmp_path / "workspace/cache/groot_inference").exists()


def test_preparation_failure_removes_only_its_partial_view(tmp_path, monkeypatch):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    snapshot(hub)
    valid = groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub)
    before = hashes(original)
    real_write = Path.write_text

    def fail_provenance(path, *args, **kwargs):
        if path.name == "inference_view.json":
            raise OSError("simulated view write failure")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_provenance)
    with pytest.raises(OSError, match="simulated view write failure"):
        groot._prepare_inference_checkpoint(original, tmp_path / "workspace", hub)
    assert list(valid.parent.iterdir()) == [valid]
    assert hashes(original) == before


def test_constructor_loads_the_view_and_keeps_original_identity(tmp_path, monkeypatch):
    original, _, _ = checkpoint(tmp_path)
    hub = tmp_path / "workspace/hub"
    snapshot(hub)
    monkeypatch.setattr(groot, "_validate_checkpoint", lambda *_: ({"original": True}, {}))
    monkeypatch.setattr(groot, "_configure_offline", lambda _: hub)
    calls = []

    class SDKPolicy:
        def __init__(self, embodiment_tag, model_path, *, device, strict=True):
            calls.append(model_path)
            assert Path(model_path) != original
            assert Path(json.loads((Path(model_path) / "config.json").read_text())["model_name"]).is_dir()
            self.model = SimpleNamespace(eval=lambda: None, action_head=SimpleNamespace(num_inference_timesteps=4))

    monkeypatch.setattr(groot, "_import_runtime", lambda _: (SimpleNamespace(manual_seed=lambda _: None), SDKPolicy))
    policy = groot.GrootPolicy(checkpoint=original, device="cpu", seed=1, workspace=tmp_path / "workspace")
    assert calls == [str(policy.inference_checkpoint)]
    assert policy.checkpoint == original
    policy.close()
    assert policy.inference_checkpoint.is_dir()
