"""Independent provenance and multi-node integration regressions; no GPU required."""

import copy
import json
import os
from pathlib import Path
import sys

import pytest

from robocasa_eval import configuration as conf, evaluation
from robocasa_eval.adapters.groot import runtime_identity


@pytest.fixture
def groot_assets(tmp_path):
    workspace = tmp_path / "xuchengxin"
    checkpoint = tmp_path / "readonly-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "Gr00tN1d7", "action_horizon": 40, "num_inference_timesteps": 4,
    }))
    (checkpoint / "model.safetensors").write_bytes(b"test-only-complete-checkpoint")
    model_name = "test/processor"
    (checkpoint / "processor_config.json").write_text(json.dumps({
        "processor_kwargs": {"model_name": model_name},
    }))
    hub = workspace / "weights" / "hub"
    cache_repo = hub / "models--test--processor"
    revision = "a" * 40
    snapshot = cache_repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (cache_repo / "refs").mkdir()
    (cache_repo / "refs" / "main").write_text(revision)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_text(json.dumps({"fixture": name, "version": 1}))
    (snapshot / "chat_template.jinja").write_text("{{ messages }}")
    (snapshot / "processing_custom.py").write_text("PROCESSOR_VERSION = 1\n")
    (snapshot / "vocabulary.bin").write_bytes(b"binary-tokenizer-vocabulary")
    (snapshot / "pytorch_model-00001-of-00004.bin").write_bytes(b"unused-base-weights")
    sdk = tmp_path / "sdk"
    package = sdk / "gr00t"
    (package / "policy").mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('provenance imported the SDK')\n")
    (package / "policy" / "gr00t_policy.py").write_text("MODEL_IMPLEMENTATION_VERSION = 1\n")
    decoder = tmp_path / "decoder.onnx"
    decoder.write_bytes(b"test-only-decoder")
    return dict(workspace=workspace, checkpoint=checkpoint, repo=sdk,
                snapshot=snapshot, cache_repo=cache_repo, decoder=decoder)


def identity(assets, *, repo=True):
    return runtime_identity(
        checkpoint=assets["checkpoint"], workspace=assets["workspace"],
        repo=assets["repo"] if repo else None,
    )


def build_groot_config(assets):
    config, _ = conf.build_config({
        "workspace": str(assets["workspace"]), "envs_per_gpu": 2,
        "decoder_path": str(assets["decoder"]), "sim_python": sys.executable,
        "policy": {"factory": "groot", "id": "groot-audit-v1", "kwargs": {
            "checkpoint": str(assets["checkpoint"]), "repo": str(assets["repo"]),
        }},
    }, verify_files=True, environment={})
    return config


def test_groot_provenance_uses_actual_cache_and_does_not_import_or_write(groot_assets):
    before_environment = dict(os.environ)
    before_files = sorted(str(p) for p in groot_assets["workspace"].rglob("*"))
    data = identity(groot_assets)
    processor = data["processor"]
    assert processor["snapshot"] == str(groot_assets["snapshot"])
    assert processor["revision"] == "a" * 40
    assert {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
            "processing_custom.py", "vocabulary.bin"} <= set(processor["files"])
    assert "pytorch_model-00001-of-00004.bin" not in processor["files"]
    assert data["implementation"]["source_files"] >= 2
    assert dict(os.environ) == before_environment
    assert sorted(str(p) for p in groot_assets["workspace"].rglob("*")) == before_files


@pytest.mark.parametrize("filename", ["tokenizer.json", "chat_template.jinja", "processing_custom.py", "vocabulary.bin"])
def test_actual_processor_assets_are_content_fingerprinted(groot_assets, filename):
    before = identity(groot_assets)
    (groot_assets["snapshot"] / filename).write_bytes(b"new processor content")
    after = identity(groot_assets)
    assert before["processor"]["sha256"] != after["processor"]["sha256"]
    assert before["implementation"] == after["implementation"]


def test_moving_processor_main_reference_invalidates_provenance(groot_assets):
    import shutil
    before = identity(groot_assets)
    new_snapshot = groot_assets["snapshot"].parent / ("b" * 40)
    shutil.copytree(groot_assets["snapshot"], new_snapshot)
    (groot_assets["cache_repo"] / "refs" / "main").write_text("b" * 40)
    after = identity(groot_assets)
    assert before["processor"]["sha256"] == after["processor"]["sha256"]
    assert before != after
    assert after["processor"]["revision"] == "b" * 40


def test_installed_groot_source_is_hashed_without_importing_package(groot_assets, monkeypatch):
    monkeypatch.syspath_prepend(str(groot_assets["repo"]))
    monkeypatch.delitem(sys.modules, "gr00t", raising=False)
    before = identity(groot_assets, repo=False)
    assert "gr00t" not in sys.modules
    (groot_assets["repo"] / "gr00t" / "policy" / "gr00t_policy.py").write_text("MODEL_IMPLEMENTATION_VERSION = 2\n")
    after = identity(groot_assets, repo=False)
    assert before["implementation"]["source_sha256"] != after["implementation"]["source_sha256"]
    assert "gr00t" not in sys.modules


def test_groot_processor_change_invalidates_real_smoke_receipt(groot_assets):
    original = build_groot_config(groot_assets)
    receipt = evaluation.smoke_receipt_path(original)
    evaluation.atomic_json(receipt, {
        "status": "passed", "full_horizons": True,
        "protocol": conf.gate_protocol(original), "tested_environment_count": 4,
        "tested_categories": ["atomic", "composite"],
    })
    assert evaluation.require_smoke(original)["status"] == "passed"
    (groot_assets["snapshot"] / "tokenizer.json").write_text('{"changed": true}')
    changed = build_groot_config(groot_assets)
    assert changed["identity"] != original["identity"]
    with pytest.raises(ValueError, match="smoke"):
        evaluation.require_smoke(changed)


def test_model_dependency_versions_invalidate_gate_without_framework_import(groot_assets, monkeypatch):
    import importlib.metadata
    versions = {"numpy": "2.2.5", "torch": "2.7.0", "transformers": "4.56.0"}

    def metadata_version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", metadata_version)
    before = build_groot_config(groot_assets)
    runtime = before["policy_identity"]["runtime"]
    assert runtime["versions"] == {
        "python": sys.version, "numpy": "2.2.5", "torch": "2.7.0",
        "transformers": "4.56.0", "tokenizers": None, "accelerate": None,
        "safetensors": None, "huggingface_hub": None,
    }
    versions["transformers"] = "4.57.0"
    after = build_groot_config(groot_assets)
    assert before["identity"] != after["identity"]
    assert conf.gate_protocol(before) != conf.gate_protocol(after)


def test_artifact_identity_covers_nested_directory_symlinks(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"weights")
    external = tmp_path / "shared-processor"
    external.mkdir()
    stats = external / "statistics.json"
    stats.write_text('{"mean": 0}')
    (root / "processor").symlink_to(external, target_is_directory=True)
    before = conf.artifact_identity(root)
    stats.write_text('{"mean": 1}')
    after = conf.artifact_identity(root)
    assert before["sha256"] != after["sha256"]


def test_artifact_directory_symlink_cycle_is_rejected(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"weights")
    (root / "recursive").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="cycle|alias|symlink|symbolic"):
        conf.artifact_identity(root)


def test_runtime_structural_assets_invalidate_identity_without_hashing_large_meshes(tmp_path):
    root = tmp_path / "robosuite_models"
    asset_dir = root / "assets" / "robots" / "g1"
    asset_dir.mkdir(parents=True)
    (root / "__init__.py").write_text("ROBOT_VERSION = 1\n")
    xml = asset_dir / "robot.xml"
    xml.write_text('<mujoco><option timestep="0.005"/></mujoco>')
    (asset_dir / "large_mesh.stl").write_bytes(b"irrelevant-large-mesh-fixture")
    before = conf.source_tree_identity(root)
    xml.write_text('<mujoco><option timestep="0.002"/></mujoco>')
    after = conf.source_tree_identity(root)
    assert before["source_sha256"] != after["source_sha256"]


def test_second_node_egl_uses_local_device_ordinals_and_one_common_run_identity(tmp_path):
    options = {"workspace": str(tmp_path / "xuchengxin"), "total_gpus": 16,
               "envs_per_gpu": 2, "renderer": "egl"}
    configs = []
    for node in (0, 1):
        config, topo = conf.build_config(options, environment={
            "NNODES": "2", "GPUS_PER_NODE": "8", "NODE_RANK": str(node),
        })
        assert topo["local"] == 8
        configs.append(config)
        for local_rank in range(8):
            env = evaluation.sim_environment(config, node * 8 + local_rank)
            assert env["MUJOCO_EGL_DEVICE_ID"] == str(local_rank)
    assert configs[0]["identity"] == configs[1]["identity"]
