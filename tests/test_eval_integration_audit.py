"""Generic artifact provenance and multi-node regressions; no GPU required."""

import sys

import pytest

from robocasa_eval import configuration as conf, evaluation


def test_declared_artifact_change_invalidates_real_smoke_receipt(tmp_path):
    artifact = tmp_path / "processor.json"
    artifact.write_text('{"scale": 1}')
    decoder = tmp_path / "decoder.onnx"
    decoder.write_bytes(b"decoder-fixture")
    options = {
        "workspace": str(tmp_path / "workspace"), "envs_per_gpu": 2,
        "decoder_path": str(decoder), "sim_python": sys.executable,
        "policy": {"artifacts": [str(artifact)]},
    }
    original, _ = conf.build_config(options, verify_files=True, environment={})
    evaluation.atomic_json(evaluation.smoke_receipt_path(original), {
        "status": "passed", "full_horizons": True,
        "protocol": conf.gate_protocol(original), "tested_environment_count": 4,
        "tested_categories": ["atomic", "composite"],
    })
    assert evaluation.require_smoke(original)["status"] == "passed"
    artifact.write_text('{"scale": 2}')
    changed, _ = conf.build_config(options, verify_files=True, environment={})
    assert changed["identity"] != original["identity"]
    with pytest.raises(ValueError, match="smoke"):
        evaluation.require_smoke(changed)


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
    options = {"workspace": str(tmp_path / "workspace"), "total_gpus": 16,
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
