"""Build or verify portable metadata for an already selected paired G1 dataset.

This standard-library tool checks metadata counts, paths, file membership and
full SHA256 identities. It does not select demonstrations, decode trajectories,
evaluate task success, modify source data, or contact a remote repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile


DEFAULT_TASK_MANIFEST = Path(__file__).with_name("tasks50.json")
CATEGORIES = {"atomic": "Atomic Tasks", "composite": "Composite Tasks"}
PAIR_FIELDS = (
    "source_episode_index", "output_episode_index", "raw_episode",
    "hdf5_path", "lerobot_data_path",
)
REQUIRED_META = {
    "meta/info.json", "meta/episodes.jsonl", "meta/episodes_stats.jsonl",
    "meta/tasks.jsonl", "meta/modality.json",
}
CAMERAS = ("ego_view", "left_wrist", "right_wrist")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ReleaseError(ValueError):
    """The input is incomplete, inconsistent, or outside the selected release."""


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def local_path(value):
    """Use cwd-relative paths, rejecting traversal and symlink components."""
    require(isinstance(value, (str, Path)) and bool(str(value)), "path must be nonempty")
    path = Path(value).expanduser()
    require(".." not in path.parts, f"parent traversal is forbidden: {path}")
    path = path.absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"symlink is forbidden: {component}")
    return path


def relative_path(value):
    require(isinstance(value, str) and bool(value), "relative path must be nonempty")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in value
        and path.as_posix() == value and value != ".",
        f"invalid relative path: {value!r}",
    )
    return path


def within(root, value):
    return local_path(root / relative_path(value))


def read_bytes(path):
    path = local_path(path)
    require(path.is_file(), f"missing regular file: {path}")
    value = path.read_bytes()
    require(value and not value.startswith(LFS_PREFIX), f"empty file or Git LFS pointer: {path}")
    return value


def read_json(path):
    try:
        return json.loads(read_bytes(path))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid JSON: {path}: {exc}") from exc


def read_jsonlines(path):
    try:
        return [json.loads(line) for line in read_bytes(path).splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid JSON Lines: {path}: {exc}") from exc


def digest(value):
    return hashlib.sha256(value).hexdigest()


def stat_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def files_in(root):
    """Enumerate all regular files without following any symlink."""
    root = local_path(root)
    require(root.is_dir(), f"missing directory: {root}")
    files = []
    for current, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            path = Path(current) / name
            require(not path.is_symlink(), f"symlink is forbidden: {path}")
            require(path.is_file() or path.is_dir(), f"nonregular file: {path}")
        files.extend(Path(current) / name for name in names)
    return sorted(files)


def load_config(path):
    value = read_json(path)
    require(isinstance(value, dict), "config must be an object")
    required = {"dataset_root", "pairings_dir", "release_id", "source_revision"}
    allowed = required | {"task_manifest", "episodes_per_task", "fps"}
    require(required <= value.keys(), f"config requires {sorted(required)}")
    require(not value.keys() - allowed, f"unknown config fields: {sorted(value.keys() - allowed)}")
    for key in ("release_id", "source_revision"):
        require(isinstance(value[key], str) and bool(value[key].strip()), f"{key} must be nonempty")
    config = dict(value)
    config["dataset_root"] = local_path(value["dataset_root"])
    config["pairings_dir"] = local_path(value["pairings_dir"])
    config["episodes_per_task"] = integer(value.get("episodes_per_task", 50), "episodes_per_task", 1)
    config["fps"] = integer(value.get("fps", 50), "fps", 1)
    custom = value.get("task_manifest") is not None
    manifest_path = local_path(value["task_manifest"] if custom else DEFAULT_TASK_MANIFEST)
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and isinstance(manifest.get("tasks"), list), "task manifest requires tasks")
    tasks = []
    names = set()
    for task in manifest["tasks"]:
        require(isinstance(task, dict), "task entry must be an object")
        name = task.get("name")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name), f"invalid task name: {name}")
        require(name not in names, f"duplicate task: {name}")
        require(task.get("category") in CATEGORIES, f"invalid category: {name}")
        require(task.get("split") in {"seen", "unseen"}, f"invalid split: {name}")
        names.add(name)
        tasks.append({key: task[key] for key in ("name", "category", "split")})
    require(any(t["split"] == "seen" for t in tasks), "manifest has no seen tasks")
    if not custom:
        counts = Counter((task["category"], task["split"]) for task in tasks)
        require(counts == {("atomic", "seen"): 25, ("composite", "seen"): 15, ("composite", "unseen"): 10}, "default manifest must describe the 40 seen / 10 unseen benchmark")
        require(config["episodes_per_task"] == 50 and config["fps"] == 50, "benchmark40 requires 50 episodes per task at 50 fps; use an explicit custom task_manifest for a sample")
    config.update(
        tasks=sorted(tasks, key=lambda task: task["name"]),
        scope="custom" if custom else "benchmark40",
        task_manifest_sha256=digest(read_bytes(manifest_path)),
        task_manifest_path=manifest_path,
    )
    return config


def expected_task_roots(config):
    roots = {}
    for task in config["tasks"]:
        if task["split"] != "seen":
            continue
        category = CATEGORIES[task["category"]]
        name = task["name"]
        roots[name] = (
            f"lerobot/{category}/robocasa_{name}_g1_3cam",
            f"hdf5/{category}/{name}_sonic",
        )
    return roots


def validate_task_directories(config, roots):
    for format_name, offset in (("lerobot", 0), ("hdf5", 1)):
        root = within(config["dataset_root"], format_name)
        require(root.is_dir(), f"missing directory: {root}")
        actual = set()
        for category in root.iterdir():
            local_path(category)
            require(category.is_dir() and category.name in CATEGORIES.values(), f"unexpected category entry: {category}")
            for task in category.iterdir():
                local_path(task)
                require(task.is_dir(), f"unexpected task entry: {task}")
                actual.add(task.relative_to(config["dataset_root"]).as_posix())
        expected = {pair[offset] for pair in roots.values()}
        require(actual == expected, f"{format_name} task set differs: missing={sorted(expected - actual)}, unexpected or unseen={sorted(actual - expected)}")
    actual_pairings = {p.relative_to(config["pairings_dir"]).as_posix() for p in files_in(config["pairings_dir"])}
    expected_pairings = {f"{name}.json" for name in roots}
    require(actual_pairings == expected_pairings, "pairings directory must contain exactly one <Task>.json for each seen task")


def indexed(records, key, count, label):
    require(isinstance(records, list) and len(records) == count, f"{label}: expected {count} records")
    require(all(isinstance(item, dict) for item in records), f"{label}: records must be objects")
    indices = [integer(record.get(key), f"{label}.{key}") for record in records]
    require(sorted(indices) == list(range(count)), f"{label}: {key} must be unique and contiguous from zero")
    return sorted(records, key=lambda row: row[key])


def format_path(pattern, **values):
    require(isinstance(pattern, str), "dataset path pattern must be a string")
    try:
        formatted = pattern.format(**values)
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise ReleaseError(f"invalid dataset path pattern: {pattern}") from exc
    return relative_path(formatted).as_posix()


def task_evidence(config, task, roots):
    name = task["name"]
    dataset = config["dataset_root"]
    n = config["episodes_per_task"]
    lerobot_relative, raw_relative = roots
    lerobot = within(dataset, lerobot_relative)
    raw = within(dataset, raw_relative)
    info = read_json(lerobot / "meta/info.json")
    require(isinstance(info, dict), f"{name}: info.json must be an object")
    require(info.get("codebase_version") == "v2.1", f"{name}: requires LeRobot v2.1 metadata")
    require(integer(info.get("total_episodes"), "total_episodes") == n, f"{name}: episode count differs")
    require(info.get("fps") == config["fps"] and type(info.get("fps")) in (int, float), f"{name}: fps differs")
    require(not info.get("discarded_episode_indices"), f"{name}: selected release still has discarded episodes")
    chunks = integer(info.get("chunks_size"), "chunks_size", 1)
    episodes = indexed(read_jsonlines(lerobot / "meta/episodes.jsonl"), "episode_index", n, name)
    frames = sum(integer(ep.get("length"), f"{name}.length", 1) for ep in episodes)
    require(integer(info.get("total_frames"), "total_frames", 1) == frames, f"{name}: frame count differs from episode lengths")
    stats = indexed(read_jsonlines(lerobot / "meta/episodes_stats.jsonl"), "episode_index", n, f"{name} statistics")
    require(all(isinstance(row.get("stats"), dict) and row["stats"] for row in stats), f"{name}: missing episode statistics")
    tasks = read_jsonlines(lerobot / "meta/tasks.jsonl")
    indexed(tasks, "task_index", integer(info.get("total_tasks"), "total_tasks", 1), f"{name} instructions")
    require(all(isinstance(row.get("task"), str) and row["task"].strip() for row in tasks), f"{name}: invalid task text")
    instructions = {row["task"] for row in tasks}
    require(all(isinstance(ep.get("tasks"), list) and ep["tasks"] and all(isinstance(text, str) and text in instructions for text in ep["tasks"]) for ep in episodes), f"{name}: episode instructions absent from tasks.jsonl")
    modality = read_json(lerobot / "meta/modality.json")
    require(isinstance(modality, dict) and modality, f"{name}: missing modality schema")
    features = info.get("features")
    require(isinstance(features, dict), f"{name}: missing features")
    for key, width in (("observation.state", 43), ("observation.projected_gravity", 3), ("action.motion_token", 64), ("teleop.left_hand_joints", 7), ("teleop.right_hand_joints", 7)):
        require(isinstance(features.get(key), dict) and features[key].get("shape") == [width], f"{name}: {key} must declare shape [{width}]")
    cameras = sorted(key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video")
    require(cameras == sorted(f"observation.images.{camera}" for camera in CAMERAS), f"{name}: requires ego_view, left_wrist and right_wrist video features")
    require(integer(info.get("total_videos"), "total_videos") == n * len(cameras), f"{name}: video count differs")
    for episode, row in zip(episodes, stats):
        for key, feature in features.items():
            require(isinstance(feature, dict), f"{name}: invalid feature {key}")
            if feature.get("dtype") in {"video", "image"}:
                continue
            values = row["stats"].get(key)
            require(isinstance(values, dict) and {"min", "max", "mean", "std", "count"} <= values.keys(), f"{name}: incomplete statistics for {key}")
            require(values["count"] == [episode["length"]], f"{name}: statistic count differs from episode length for {key}")
    for key, start, end in (("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15), ("left_arm", 15, 22), ("left_hand", 22, 29), ("right_arm", 29, 36), ("right_hand", 36, 43)):
        value = modality.get("state", {}).get(key, {})
        require(value.get("start") == start and value.get("end") == end and value.get("original_key", "observation.state") == "observation.state", f"{name}: inconsistent state modality {key}")
    for section, key, original, width in (("state", "projected_gravity", "observation.projected_gravity", 3), ("action", "motion_token", "action.motion_token", 64), ("action", "left_hand_joints", "teleop.left_hand_joints", 7), ("action", "right_hand_joints", "teleop.right_hand_joints", 7)):
        value = modality.get(section, {}).get(key, {})
        require(value.get("start") == 0 and value.get("end") == width and value.get("original_key") == original, f"{name}: inconsistent modality {section}.{key}")
    for camera in CAMERAS:
        require(modality.get("video", {}).get(camera, {}).get("original_key") == f"observation.images.{camera}", f"{name}: missing video modality {camera}")
    require(modality.get("annotation", {}).get("human.task_description", {}).get("original_key") == "task_index", f"{name}: missing instruction modality")
    pairing_path = config["pairings_dir"] / f"{name}.json"
    pairings = indexed(read_json(pairing_path), "output_episode_index", n, f"{name} pairings")
    source_ids, raw_ids = set(), set()
    selected = []
    expected_data, expected_videos = set(), set()
    for episode, pairing in zip(episodes, pairings):
        source_id = integer(pairing.get("source_episode_index"), "source_episode_index")
        raw_id = pairing.get("raw_episode")
        require(source_id not in source_ids, f"{name}: duplicate source episode index")
        require(isinstance(raw_id, str) and re.fullmatch(r"ep_[A-Za-z0-9_.-]+", raw_id), f"{name}: invalid raw episode name")
        require(raw_id not in raw_ids, f"{name}: duplicate raw episode")
        source_ids.add(source_id)
        raw_ids.add(raw_id)
        index = episode["episode_index"]
        values = {"episode_index": index, "episode_chunk": index // chunks}
        data_path = format_path(info.get("data_path"), **values)
        require(data_path.startswith("data/") and data_path.endswith(".parquet"), f"{name}: invalid data path")
        require(data_path not in expected_data, f"{name}: data paths are not unique")
        expected_data.add(data_path)
        for key in cameras:
            video = format_path(info.get("video_path"), video_key=key, **values)
            require(video.startswith("videos/") and video.endswith(".mp4"), f"{name}: invalid video path")
            require(video not in expected_videos, f"{name}: video paths are not unique")
            expected_videos.add(video)
        hdf5 = f"{raw_relative}/episodes/{raw_id}/ep_demo.hdf5"
        parquet = f"{lerobot_relative}/{data_path}"
        require(relative_path(pairing.get("hdf5_path")).as_posix() == hdf5, f"{name}: HDF5 pairing path differs")
        require(relative_path(pairing.get("lerobot_data_path")).as_posix() == parquet, f"{name}: LeRobot pairing path differs")
        require(within(dataset, hdf5).is_file(), f"{name}: missing paired HDF5: {hdf5}")
        selected.append({key: pairing[key] for key in PAIR_FIELDS})
    lerobot_files = files_in(lerobot)
    relative_files = {path.relative_to(lerobot).as_posix() for path in lerobot_files}
    require(REQUIRED_META <= relative_files, f"{name}: incomplete required metadata")
    require({p for p in relative_files if p.startswith("data/")} == expected_data, f"{name}: data file set differs")
    require({p for p in relative_files if p.startswith("videos/")} == expected_videos, f"{name}: video file set differs")
    require(all(p.startswith(("data/", "videos/", "meta/")) for p in relative_files), f"{name}: unexpected file outside data/videos/meta")
    require({p.name for p in raw.iterdir()} == {"episodes"}, f"{name}: unexpected raw root entries")
    episodes_dir = within(raw, "episodes")
    require(episodes_dir.is_dir(), f"{name}: missing raw episodes directory")
    require({p.name for p in episodes_dir.iterdir()} == raw_ids, f"{name}: raw episode file set differs")
    require(all(p.is_dir() and not p.is_symlink() for p in episodes_dir.iterdir()), f"{name}: invalid raw episode directory")
    raw_files = files_in(raw)
    summary = {
        "task": name, "category": task["category"], "episodes": n,
        "frames": frames, "videos": len(expected_videos),
        "lerobot_root": lerobot_relative, "hdf5_root": raw_relative,
        "episode_mapping": f"episode_mapping/{name}.json",
        "checksums": f"checksums/{name}.json",
        "pairing_input_sha256": digest(read_bytes(pairing_path)),
    }
    return summary, selected, sorted(lerobot_files + raw_files)


def file_identity(path):
    local_path(path)
    before = stat_identity(path)
    require(before[2] > 0, f"empty data file: {path}")
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        first = stream.read(8192)
        require(not first.startswith(LFS_PREFIX), f"Git LFS pointer is not data: {path}")
        if path.suffix == ".hdf5":
            require(first.startswith(b"\x89HDF\r\n\x1a\n"), f"invalid HDF5 signature: {path}")
        if path.suffix == ".parquet":
            require(first.startswith(b"PAR1"), f"invalid Parquet signature: {path}")
        hasher.update(first)
        size += len(first)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
    after = stat_identity(path)
    require(before == after and size == before[2], f"file changed while hashing: {path}")
    return {"bytes": size, "sha256": hasher.hexdigest()}, after


def collect(config):
    roots = expected_task_roots(config)
    validate_task_directories(config, roots)
    payloads, summaries, snapshots = {}, [], {}
    all_data_files = set()
    for task in config["tasks"]:
        if task["split"] != "seen":
            continue
        name = task["name"]
        initial_files = sorted(files_in(within(config["dataset_root"], roots[name][0])) + files_in(within(config["dataset_root"], roots[name][1])))
        initial_stats = {path: stat_identity(path) for path in initial_files}
        pairing_path = config["pairings_dir"] / f"{name}.json"
        pairing_stat = stat_identity(pairing_path)
        summary, mapping, paths = task_evidence(config, task, roots[name])
        all_data_files.update(paths)
        require(paths == initial_files, f"{name}: file set changed during validation")
        hashes = []
        for path in paths:
            identity, snapshot = file_identity(path)
            hashes.append({"path": path.relative_to(config["dataset_root"]).as_posix(), **identity})
            require(snapshot == initial_stats[path], f"file changed during validation: {path}")
            snapshots[path] = snapshot
        require(paths == sorted(files_in(within(config["dataset_root"], roots[name][0])) + files_in(within(config["dataset_root"], roots[name][1]))), f"{name}: file set changed while hashing")
        summary.update(files=len(hashes), bytes=sum(row["bytes"] for row in hashes))
        summaries.append(summary)
        require(stat_identity(pairing_path) == pairing_stat, f"{name}: input pairing changed during validation")
        snapshots[pairing_path] = pairing_stat
        payloads[summary["episode_mapping"]] = mapping
        payloads[summary["checksums"]] = {"schema_version": 1, "task": name, "algorithm": "sha256", "path_base": "dataset_root", "files": hashes}
    validate_task_directories(config, roots)
    final_data_files = set()
    for pair in roots.values():
        for root in pair:
            final_data_files.update(files_in(within(config["dataset_root"], root)))
    require(final_data_files == all_data_files, "data file set changed during metadata collection")
    require(all(stat_identity(path) == before for path, before in snapshots.items()), "data changed during metadata collection")
    require(digest(read_bytes(config["task_manifest_path"])) == config["task_manifest_sha256"], "task manifest changed during metadata collection")
    payloads["release_manifest.json"] = {
        "schema_version": 1,
        "format": "robocasa-g1-paired-release",
        "scope": config["scope"],
        "release_id": config["release_id"],
        "source_revision": config["source_revision"],
        "source_revision_kind": "caller_supplied_label",
        "task_manifest_sha256": config["task_manifest_sha256"],
        "episodes_per_task": config["episodes_per_task"],
        "fps": config["fps"],
        "metadata_path_base": "metadata_root",
        "data_path_base": "dataset_root",
        "frame_count_basis": "episodes_jsonl_lengths",
        "total_tasks": len(summaries),
        "total_episodes": sum(row["episodes"] for row in summaries),
        "total_frames": sum(row["frames"] for row in summaries),
        "total_videos": sum(row["videos"] for row in summaries),
        "total_files": sum(row["files"] for row in summaries),
        "total_bytes": sum(row["bytes"] for row in summaries),
        "unseen_tasks_without_data": [task["name"] for task in config["tasks"] if task["split"] == "unseen"],
        "tasks": summaries,
    }
    return payloads


def output_path(config, value):
    output = local_path(value)
    for source in (config["dataset_root"], config["pairings_dir"]):
        require(output != source and source not in output.parents and output not in source.parents, f"metadata output must not overlap source inputs: {output}")
    return output


def build(config, output):
    output = output_path(config, output)
    require(not output.exists(), f"refusing to overwrite metadata: {output}")
    payloads = collect(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for relative, value in payloads.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        require(not output.exists() and not output.is_symlink(), f"metadata appeared during build: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return payloads["release_manifest.json"]


def verify(config, output):
    output = output_path(config, output)
    expected = collect(config)
    actual = {path.relative_to(output).as_posix() for path in files_in(output)}
    require(actual == set(expected), f"metadata file set differs: missing={sorted(set(expected) - actual)}, unexpected={sorted(actual - set(expected))}")
    for relative, value in expected.items():
        require(canonical(read_json(within(output, relative))) == canonical(value), f"metadata or source identity differs: {relative}")
    return expected["release_manifest.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New metadata directory for build; existing directory for verify.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = (build if args.command == "build" else verify)(config, args.output)
    except (ReleaseError, OSError, TypeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{args.command} complete: scope={result['scope']} tasks={result['total_tasks']} episodes={result['total_episodes']} files={result['total_files']} bytes={result['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
