"""Release metadata checks use synthetic files, never production datasets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from robocasa_eval import dataset_release as release


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_jsonlines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def sample(tmp_path):
    dataset = tmp_path / "data"
    pairings = tmp_path / "pairings"
    task_manifest = tmp_path / "tasks.json"
    write_json(task_manifest, {"tasks": [
        {"name": "CloseFridge", "category": "atomic", "split": "seen"},
        {"name": "HoldOut", "category": "composite", "split": "unseen"},
    ]})
    lerobot = "lerobot/Atomic Tasks/robocasa_CloseFridge_g1_3cam"
    raw = "hdf5/Atomic Tasks/CloseFridge_sonic"
    root = dataset / lerobot
    features = {key: {"dtype": "float32", "shape": [size]} for key, size in [
        ("observation.state", 43), ("observation.projected_gravity", 3),
        ("action.motion_token", 64), ("teleop.left_hand_joints", 7),
        ("teleop.right_hand_joints", 7),
    ]}
    for camera in release.CAMERAS:
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [16, 16, 3]}
    write_json(root / "meta/info.json", {
        "codebase_version": "v2.1", "total_episodes": 2, "total_frames": 5,
        "total_tasks": 1, "total_videos": 6, "fps": 50, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    })
    modality = {
        "state": {key: {"start": start, "end": end} for key, start, end in [
            ("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15),
            ("left_arm", 15, 22), ("left_hand", 22, 29),
            ("right_arm", 29, 36), ("right_hand", 36, 43),
        ]},
        "action": {key: {"start": 0, "end": width, "original_key": original} for key, width, original in [
            ("motion_token", 64, "action.motion_token"),
            ("left_hand_joints", 7, "teleop.left_hand_joints"),
            ("right_hand_joints", 7, "teleop.right_hand_joints"),
        ]},
        "video": {key: {"original_key": f"observation.images.{key}"} for key in release.CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    modality["state"]["projected_gravity"] = {"start": 0, "end": 3, "original_key": "observation.projected_gravity"}
    write_json(root / "meta/modality.json", modality)
    write_jsonlines(root / "meta/episodes.jsonl", [
        {"episode_index": index, "length": index + 2, "tasks": ["Close the fridge."]}
        for index in range(2)
    ])
    write_jsonlines(root / "meta/episodes_stats.jsonl", [
        {"episode_index": index, "stats": {
            key: {**{stat: [0] * feature["shape"][0] for stat in ("min", "max", "mean", "std")}, "count": [index + 2]}
            for key, feature in features.items() if feature["dtype"] != "video"
        }}
        for index in range(2)
    ])
    write_jsonlines(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "Close the fridge."}])
    mappings = []
    for index in range(2):
        raw_episode = f"ep_{1000 + index}"
        parquet = f"{lerobot}/data/chunk-000/episode_{index:06d}.parquet"
        hdf5 = f"{raw}/episodes/{raw_episode}/ep_demo.hdf5"
        videos = [(f"{lerobot}/videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4", b"synthetic video") for camera in release.CAMERAS]
        for relative, payload in [(parquet, b"PAR1-synthetic-PAR1"), (hdf5, b"\x89HDF\r\n\x1a\n-synthetic"), *videos]:
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        mappings.append({
            "source_episode_index": index + 7, "output_episode_index": index,
            "raw_episode": raw_episode, "hdf5_path": hdf5, "lerobot_data_path": parquet,
            "success": True, "private_source": "/private/recordings",
        })
    write_json(pairings / "CloseFridge.json", mappings)
    config_path = tmp_path / "config.json"
    write_json(config_path, {
        "dataset_root": str(dataset), "pairings_dir": str(pairings),
        "task_manifest": str(task_manifest), "release_id": "sample-validation",
        "source_revision": "local-fixture", "episodes_per_task": 2,
    })
    return {
        "dataset": dataset, "pairings": pairings, "lerobot": root,
        "raw": dataset / raw, "config_path": config_path,
        "output": tmp_path / "release", "task_manifest": task_manifest,
    }


def config(sample):
    return release.load_config(sample["config_path"])


def snapshots(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def test_build_and_verify_all_files_without_mutating_sources(sample):
    before = snapshots(sample["dataset"])
    result = release.build(config(sample), sample["output"])
    assert result["scope"] == "custom"
    assert result["total_tasks"] == 1
    assert result["total_episodes"] == 2
    assert result["total_frames"] == 5
    assert result["total_files"] == len(before)
    assert result["unseen_tasks_without_data"] == ["HoldOut"]
    assert result["source_revision_kind"] == "caller_supplied_label"
    rows = json.loads((sample["output"] / "episode_mapping/CloseFridge.json").read_text())
    assert all(set(row) == set(release.PAIR_FIELDS) for row in rows)
    assert "private_source" not in (sample["output"] / "episode_mapping/CloseFridge.json").read_text()
    assert release.verify(config(sample), sample["output"]) == result
    assert snapshots(sample["dataset"]) == before


def test_verification_is_portable_after_moving_inputs_and_metadata(sample, tmp_path):
    release.build(config(sample), sample["output"])
    moved = tmp_path / "moved_data"
    sample["dataset"].rename(moved)
    mutate_json(sample["config_path"], lambda v: v.update(dataset_root=str(moved)))
    moved_output = tmp_path / "moved_metadata"
    sample["output"].rename(moved_output)
    assert release.verify(config(sample), moved_output)["total_episodes"] == 2


@pytest.mark.parametrize("change", ["modify", "missing", "extra", "metadata"])
def test_verify_detects_source_changes(sample, change):
    release.build(config(sample), sample["output"])
    parquet = sample["lerobot"] / "data/chunk-000/episode_000000.parquet"
    if change == "modify":
        parquet.write_bytes(b"PAR1-other-payload-PAR1")
    elif change == "missing":
        parquet.unlink()
    elif change == "extra":
        parquet.with_name("extra.parquet").write_bytes(b"PAR1-extra")
    else:
        mutate_json(sample["lerobot"] / "meta/info.json", lambda v: v.update(total_frames=6))
    metadata_before = snapshots(sample["output"])
    with pytest.raises(release.ReleaseError):
        release.verify(config(sample), sample["output"])
    assert snapshots(sample["output"]) == metadata_before


@pytest.mark.parametrize("name", ["release_manifest.json", "checksums/CloseFridge.json", "episode_mapping/CloseFridge.json"])
def test_verify_does_not_trust_modified_metadata(sample, name):
    release.build(config(sample), sample["output"])
    path = sample["output"] / name
    value = json.loads(path.read_text())
    if isinstance(value, list):
        value[0]["source_episode_index"] = 99
    elif name.startswith("checksums"):
        value["files"][0]["sha256"] = "0" * 64
    else:
        value["total_frames"] = 99
    write_json(path, value)
    with pytest.raises(release.ReleaseError, match="identity differs"):
        release.verify(config(sample), sample["output"])


@pytest.mark.parametrize("field", ["source_episode_index", "output_episode_index", "raw_episode"])
def test_duplicate_pairings_fail_before_creating_outputs(sample, field):
    mutate_json(sample["pairings"] / "CloseFridge.json", lambda rows: rows[1].update({field: rows[0][field]}))
    with pytest.raises(release.ReleaseError):
        release.build(config(sample), sample["output"])
    assert not sample["output"].exists()
    assert not list(sample["output"].parent.glob(".release.partial-*"))


@pytest.mark.parametrize("field,path", [
    ("hdf5_path", "../escape.hdf5"), ("lerobot_data_path", "/absolute.parquet"),
    ("hdf5_path", "hdf5/Atomic Tasks/CloseFridge_sonic/episodes/ep_1001/ep_demo.hdf5"),
])
def test_pairing_paths_cannot_escape_or_point_to_wrong_episode(sample, field, path):
    mutate_json(sample["pairings"] / "CloseFridge.json", lambda rows: rows[0].update({field: path}))
    with pytest.raises(release.ReleaseError):
        release.build(config(sample), sample["output"])


@pytest.mark.parametrize("where", ["hdf5", "lerobot", "pairings"])
def test_extra_tasks_or_unseen_data_are_rejected(sample, where):
    if where == "pairings":
        write_json(sample["pairings"] / "HoldOut.json", [])
    else:
        name = "HoldOut_sonic" if where == "hdf5" else "robocasa_HoldOut_g1_3cam"
        (sample["dataset"] / where / "Composite Tasks" / name).mkdir(parents=True)
    with pytest.raises(release.ReleaseError):
        release.build(config(sample), sample["output"])


def test_missing_seen_task_is_rejected(sample):
    shutil.rmtree(sample["raw"])
    with pytest.raises(release.ReleaseError, match="task set differs"):
        release.build(config(sample), sample["output"])


@pytest.mark.parametrize("entry", ["data", "input_root", "output"])
def test_symlinks_are_rejected(sample, entry, tmp_path):
    if entry == "data":
        target = sample["lerobot"] / "data/chunk-000/episode_000000.parquet"
        copy = tmp_path / "original.parquet"
        target.rename(copy)
        target.symlink_to(copy)
    elif entry == "input_root":
        link = tmp_path / "dataset-link"
        link.symlink_to(sample["dataset"], target_is_directory=True)
        mutate_json(sample["config_path"], lambda v: v.update(dataset_root=str(link)))
    else:
        sample["output"].symlink_to(tmp_path / "absent")
    with pytest.raises(release.ReleaseError, match="symlink"):
        release.build(config(sample), sample["output"])


@pytest.mark.parametrize("kind", ["lfs", "empty", "wrong_magic"])
def test_missing_downloaded_payload_is_rejected(sample, kind):
    path = sample["raw"] / "episodes/ep_1000/ep_demo.hdf5"
    payload = {"lfs": release.LFS_PREFIX + b"\noid sha256:000\nsize 10000\n", "empty": b"", "wrong_magic": b"not hdf5"}[kind]
    path.write_bytes(payload)
    with pytest.raises(release.ReleaseError):
        release.build(config(sample), sample["output"])


@pytest.mark.parametrize("where", ["dataset", "pairings"])
def test_output_cannot_overlap_inputs(sample, where):
    with pytest.raises(release.ReleaseError, match="overlap"):
        release.build(config(sample), sample[where] / "metadata")


def test_existing_output_is_never_overwritten(sample):
    sample["output"].mkdir()
    keep = sample["output"] / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(release.ReleaseError, match="overwrite"):
        release.build(config(sample), sample["output"])
    assert keep.read_text() == "keep"


def test_default_manifest_enforces_fifty_episode_gate(sample):
    mutate_json(sample["config_path"], lambda v: v.pop("task_manifest"))
    with pytest.raises(release.ReleaseError, match="50 episodes"):
        config(sample)


def test_write_failure_cleans_only_own_stage(sample, monkeypatch):
    keep = sample["output"].parent / ".release.partial-other"
    keep.mkdir()
    original = Path.write_text

    def failed_write(path, *args, **kwargs):
        if any(part.startswith(".release.partial-") for part in path.parts):
            raise OSError("disk write failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failed_write)
    with pytest.raises(OSError, match="disk write failed"):
        release.build(config(sample), sample["output"])
    assert not sample["output"].exists()
    assert list(sample["output"].parent.glob(".release.partial-*")) == [keep]


def test_cli_uses_only_standard_library(sample):
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable, "-B", "-S", "-m", "robocasa_eval.dataset_release"]
    for command in ["--help", "build", "verify"]:
        arguments = [command] if command == "--help" else [command, "--config", str(sample["config_path"]), "--output", str(sample["output"])]
        result = subprocess.run(base + arguments, cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_verify_rejects_extra_metadata_files(sample):
    release.build(config(sample), sample["output"])
    (sample["output"] / "untracked.json").write_text("{}")
    with pytest.raises(release.ReleaseError, match="metadata file set differs"):
        release.verify(config(sample), sample["output"])


@pytest.mark.parametrize("change", ["stats_count", "missing_stats", "modality", "feature_shape"])
def test_incomplete_or_inconsistent_metadata_is_rejected(sample, change):
    meta = sample["lerobot"] / "meta"
    if change in {"stats_count", "missing_stats"}:
        rows = release.read_jsonlines(meta / "episodes_stats.jsonl")
        if change == "stats_count":
            rows[0]["stats"]["action.motion_token"]["count"] = [100]
        else:
            del rows[0]["stats"]["action.motion_token"]
        write_jsonlines(meta / "episodes_stats.jsonl", rows)
    elif change == "modality":
        mutate_json(meta / "modality.json", lambda value: value["state"]["left_hand"].update(start=29))
    else:
        mutate_json(meta / "info.json", lambda value: value["features"]["action.motion_token"].update(shape=[43]))
    with pytest.raises(release.ReleaseError):
        release.build(config(sample), sample["output"])
    assert not sample["output"].exists()


def test_existing_hf_pairings_location_inside_dataset_is_supported(sample):
    moved = sample["dataset"] / "metadata/old-release/episode_mapping"
    moved.parent.mkdir(parents=True)
    sample["pairings"].rename(moved)
    mutate_json(sample["config_path"], lambda value: value.update(pairings_dir=str(moved)))
    assert release.build(config(sample), sample["output"])["total_episodes"] == 2
    release.verify(config(sample), sample["output"])


@pytest.mark.parametrize("change", ["metadata", "extra_file", "pairing"])
def test_source_changes_during_hashing_abort_before_output(sample, monkeypatch, change):
    original = release.file_identity
    changed = False

    def mutating_identity(path):
        nonlocal changed
        value = original(path)
        if not changed:
            changed = True
            if change == "metadata":
                mutate_json(sample["lerobot"] / "meta/info.json", lambda obj: obj.update(total_frames=8))
            elif change == "extra_file":
                (sample["lerobot"] / "data/new.parquet").write_bytes(b"PAR1-new")
            else:
                mutate_json(sample["pairings"] / "CloseFridge.json", lambda rows: rows[0].update(private_source="changed"))
        return value

    monkeypatch.setattr(release, "file_identity", mutating_identity)
    with pytest.raises(release.ReleaseError, match="changed"):
        release.build(config(sample), sample["output"])
    assert not sample["output"].exists()
