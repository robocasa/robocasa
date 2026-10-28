#!/usr/bin/env python3
"""Safely clean SONIC LeRobot v2.1 datasets.

The cleaner removes episodes explicitly marked as discarded and removes stale
PICO pose frames only while ``teleop.stream_mode`` is POSE or POSE_PAUSE.  It
intentionally preserves zero-SMPL frames from PLANNER and OFF modes because
those zeros have valid mode-specific semantics.

For a structurally incomplete recording, a caller may explicitly exclude a
source episode without mutating the source metadata.  This exception is
restricted to a single input dataset and recorded separately in the cleanup
report.

Outputs are built and verified in a sibling partial directory.  A completed
dataset is promoted to ``<source>_cleaned`` only after all metadata, parquet,
video, and statistics checks pass.  Existing outputs can be rotated to a
timestamped backup with ``--replace-existing``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

import av
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("clean_sonic_dataset")

DEFAULT_OUTPUT_SUFFIX = "_cleaned"
DEFAULT_POSE_MODES = (1, 4)  # POSE, POSE_PAUSE
SMPL_POSE_COLUMN = "teleop.smpl_pose"
STREAM_MODE_COLUMN = "teleop.stream_mode"


@dataclass(frozen=True)
class DatasetJob:
    """A source dataset and its non-destructive output locations."""

    source: Path
    output: Path
    partial: Path


@dataclass
class DatasetScan:
    """Read-only cleanup prediction for one source dataset."""

    source: str
    output: str
    source_episodes: int
    source_frames: int
    discarded_episodes: int
    discarded_frames: int
    extra_discarded_episode_indices: list[int]
    extra_discarded_episodes: int
    extra_discarded_frames: int
    kept_episodes_before_pose_cleanup: int
    kept_frames_before_pose_cleanup: int
    zero_smpl_by_stream_mode: dict[str, int]
    pose_zero_frames: int
    frozen_pose_leadin_frames: int
    pose_frames_to_remove: int
    episodes_with_pose_stale: int
    episodes_dropped_as_all_pose_stale: int
    expected_output_episodes: int
    expected_output_frames: int


def read_jsonlines(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records from *path*."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_info(dataset_path: Path) -> dict[str, Any]:
    """Load a dataset's ``meta/info.json``."""

    return json.loads(
        (dataset_path / "meta" / "info.json").read_text(encoding="utf-8")
    )


def get_parquet_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    """Resolve a LeRobot episode parquet path from dataset metadata."""

    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_path / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def get_video_keys(info: dict[str, Any]) -> list[str]:
    """Return feature names stored as episode video files."""

    explicit_keys = info.get("video_keys")
    if explicit_keys:
        return list(explicit_keys)
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def get_video_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    """Resolve a LeRobot episode video path from dataset metadata."""

    pattern = info.get(
        "video_path",
        "videos/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_path / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def build_mode_aware_stale_mask(
    smpl_pose: np.ndarray,
    stream_mode: np.ndarray,
    pose_modes: Sequence[int] = DEFAULT_POSE_MODES,
) -> np.ndarray:
    """Return frames to remove for stale SMPL in pose-driven modes only.

    All-zero SMPL rows are stale only in POSE / POSE_PAUSE.  A run of repeated
    non-zero pose rows immediately preceding such a zero row is also marked as
    a frozen lead-in.  Zero rows in PLANNER and OFF modes are preserved.
    """

    smpl_array = np.asarray(smpl_pose)
    mode_array = np.asarray(stream_mode).reshape(-1)
    if smpl_array.ndim != 2:
        raise ValueError(f"SMPL pose must be 2-D, got shape {smpl_array.shape}")
    if len(smpl_array) != len(mode_array):
        raise ValueError(
            "SMPL pose and stream mode lengths differ: "
            f"{len(smpl_array)} != {len(mode_array)}"
        )

    is_pose_mode = np.isin(mode_array, np.asarray(tuple(pose_modes)))
    is_zero = np.all(smpl_array == 0, axis=1)
    pose_zero = is_zero & is_pose_mode
    remove = pose_zero.copy()

    same_as_previous = np.zeros(len(smpl_array), dtype=bool)
    if len(smpl_array) > 1:
        same_as_previous[1:] = (
            np.all(smpl_array[1:] == smpl_array[:-1], axis=1)
            & (mode_array[1:] == mode_array[:-1])
        )

    for zero_index in np.flatnonzero(pose_zero):
        zero_mode = mode_array[zero_index]
        previous = int(zero_index) - 1
        while (
            previous >= 0
            and mode_array[previous] == zero_mode
            and not is_zero[previous]
            and same_as_previous[previous]
        ):
            remove[previous] = True
            previous -= 1

    return remove


def validate_source(dataset_path: Path) -> None:
    """Validate the minimum source structure required by the cleaner."""

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_path}")
    for relative_path in ("meta/info.json", "meta/episodes.jsonl"):
        path = dataset_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file is missing: {path}")

    info = load_info(dataset_path)
    features = info.get("features", {})
    for feature_name in (SMPL_POSE_COLUMN, STREAM_MODE_COLUMN):
        if feature_name not in features:
            raise ValueError(
                f"dataset {dataset_path} has no required feature {feature_name!r}"
            )


def scan_dataset(
    job: DatasetJob,
    pose_modes: Sequence[int],
    extra_discarded: Sequence[int] = (),
) -> DatasetScan:
    """Predict discard and mode-aware pose cleanup without writing files."""

    info = load_info(job.source)
    episodes = read_jsonlines(job.source / "meta" / "episodes.jsonl")
    discarded = set(int(index) for index in info.get("discarded_episode_indices", []))
    requested_extra = set(int(index) for index in extra_discarded)
    episode_indices = {int(episode["episode_index"]) for episode in episodes}
    unknown_extra = requested_extra - episode_indices
    if unknown_extra:
        raise ValueError(
            f"extra discard episode indices are not present in {job.source}: "
            f"{sorted(unknown_extra)}"
        )
    overlapping_extra = requested_extra & discarded
    if overlapping_extra:
        raise ValueError(
            "extra discard episode indices are already marked discarded in "
            f"{job.source}: {sorted(overlapping_extra)}"
        )
    effective_extra = requested_extra
    all_discarded = discarded | effective_extra

    source_frames = 0
    discarded_frames = 0
    extra_discarded_frames = 0
    kept_frames = 0
    pose_zero_frames = 0
    frozen_frames = 0
    affected_episodes = 0
    all_stale_episodes = 0
    zero_by_mode: Counter[int] = Counter()

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        source_frames += episode_length
        if episode_index in discarded:
            discarded_frames += episode_length
            continue
        if episode_index in effective_extra:
            extra_discarded_frames += episode_length
            continue

        parquet_path = get_parquet_path(job.source, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing source parquet: {parquet_path}")
        for video_key in get_video_keys(info):
            video_path = get_video_path(job.source, info, episode_index, video_key)
            if not video_path.is_file():
                raise FileNotFoundError(f"missing source video: {video_path}")
        frame = pd.read_parquet(
            parquet_path,
            columns=[SMPL_POSE_COLUMN, STREAM_MODE_COLUMN],
        )
        if len(frame) != episode_length:
            raise ValueError(
                f"episode {episode_index} metadata says {episode_length} frames, "
                f"but parquet contains {len(frame)}"
            )

        smpl_pose = np.vstack(
            [np.asarray(value, dtype=np.float32) for value in frame[SMPL_POSE_COLUMN]]
        )
        stream_mode = frame[STREAM_MODE_COLUMN].to_numpy()
        is_zero = np.all(smpl_pose == 0, axis=1)
        for mode, count in zip(*np.unique(stream_mode[is_zero], return_counts=True)):
            zero_by_mode[int(mode)] += int(count)

        mask = build_mode_aware_stale_mask(smpl_pose, stream_mode, pose_modes)
        pose_zero = is_zero & np.isin(stream_mode, np.asarray(tuple(pose_modes)))
        removed = int(mask.sum())
        zero_count = int(pose_zero.sum())
        pose_zero_frames += zero_count
        frozen_frames += removed - zero_count
        affected_episodes += int(removed > 0)
        all_stale_episodes += int(removed == len(frame))
        kept_frames += len(frame)

    discarded_present = {
        int(episode["episode_index"])
        for episode in episodes
        if int(episode["episode_index"]) in discarded
    }
    extra_discarded_present = episode_indices & effective_extra
    pose_frames_to_remove = pose_zero_frames + frozen_frames
    kept_episode_count = len(episodes) - len(
        episode_indices & all_discarded
    )
    return DatasetScan(
        source=str(job.source),
        output=str(job.output),
        source_episodes=len(episodes),
        source_frames=source_frames,
        discarded_episodes=len(discarded_present),
        discarded_frames=discarded_frames,
        extra_discarded_episode_indices=sorted(extra_discarded_present),
        extra_discarded_episodes=len(extra_discarded_present),
        extra_discarded_frames=extra_discarded_frames,
        kept_episodes_before_pose_cleanup=kept_episode_count,
        kept_frames_before_pose_cleanup=kept_frames,
        zero_smpl_by_stream_mode={
            str(mode): count for mode, count in sorted(zero_by_mode.items())
        },
        pose_zero_frames=pose_zero_frames,
        frozen_pose_leadin_frames=frozen_frames,
        pose_frames_to_remove=pose_frames_to_remove,
        episodes_with_pose_stale=affected_episodes,
        episodes_dropped_as_all_pose_stale=all_stale_episodes,
        expected_output_episodes=kept_episode_count - all_stale_episodes,
        expected_output_frames=kept_frames - pose_frames_to_remove,
    )


def load_processor(processor_path: Path) -> ModuleType:
    """Load the existing GEAR-SONIC processor for format-preserving writes."""

    if not processor_path.is_file():
        raise FileNotFoundError(f"GEAR-SONIC processor not found: {processor_path}")
    module_name = "_gear_sonic_process_dataset"
    specification = importlib.util.spec_from_file_location(module_name, processor_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load processor module: {processor_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def metadata_hashes(dataset_path: Path) -> dict[str, str]:
    """Hash source metadata files to prove the input was not rewritten."""

    hashes: dict[str, str] = {}
    for name in (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "modality.json",
    ):
        path = dataset_path / "meta" / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def remove_extra_discarded_episodes(
    processed_episodes: Sequence[dict[str, Any]],
    extra_discarded: Sequence[int],
) -> list[dict[str, Any]]:
    """Exclude explicitly requested source episodes from processor output."""

    extra_indices = set(int(index) for index in extra_discarded)
    return [
        episode
        for episode in processed_episodes
        if int(episode["episode_meta"]["episode_index"]) not in extra_indices
    ]


def validate_processed_episode_indices(
    dataset_path: Path,
    processed_episodes: Sequence[dict[str, Any]],
    extra_discarded: Sequence[int],
) -> None:
    """Require the processor to return every expected source episode exactly once."""

    info = load_info(dataset_path)
    episodes = read_jsonlines(dataset_path / "meta" / "episodes.jsonl")
    metadata_indices = {int(episode["episode_index"]) for episode in episodes}
    discarded_indices = {
        int(index) for index in info.get("discarded_episode_indices", [])
    }
    extra_indices = {int(index) for index in extra_discarded}
    expected_indices = metadata_indices - discarded_indices - extra_indices
    actual_index_list = [
        int(episode["episode_meta"]["episode_index"])
        for episode in processed_episodes
    ]
    actual_indices = set(actual_index_list)

    duplicate_indices = sorted(
        index
        for index, count in Counter(actual_index_list).items()
        if count > 1
    )
    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)
    if duplicate_indices or missing_indices or unexpected_indices:
        raise RuntimeError(
            f"processor episode set mismatch for {dataset_path}: "
            f"missing={missing_indices}, unexpected={unexpected_indices}, "
            f"duplicates={duplicate_indices}"
        )


def apply_pose_cleanup(
    processed_episodes: list[dict[str, Any]],
    pose_modes: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Apply the mode-aware mask to processor episode records in memory."""

    cleaned_episodes: list[dict[str, Any]] = []
    episode_reports: list[dict[str, int]] = []
    for episode in processed_episodes:
        frame: pd.DataFrame = episode["df"]
        source_episode_index = int(episode["episode_meta"]["episode_index"])
        smpl_pose = np.vstack(
            [np.asarray(value, dtype=np.float32) for value in frame[SMPL_POSE_COLUMN]]
        )
        stream_mode = frame[STREAM_MODE_COLUMN].to_numpy()
        mask = build_mode_aware_stale_mask(smpl_pose, stream_mode, pose_modes)
        removed = int(mask.sum())
        pose_zero = np.all(smpl_pose == 0, axis=1) & np.isin(
            stream_mode,
            np.asarray(tuple(pose_modes)),
        )
        zero_count = int(pose_zero.sum())

        if removed == len(frame):
            episode_reports.append(
                {
                    "source_episode_index": source_episode_index,
                    "output_episode_index": -1,
                    "source_frames": len(frame),
                    "output_frames": 0,
                    "pose_zero_frames_removed": zero_count,
                    "frozen_pose_leadin_frames_removed": removed - zero_count,
                }
            )
            continue

        if removed:
            valid_indices = np.flatnonzero(~mask)
            episode["df"] = frame.iloc[valid_indices].copy().reset_index(drop=True)
            episode["valid_indices"] = valid_indices

        output_episode_index = len(cleaned_episodes)
        cleaned_episodes.append(episode)
        episode_reports.append(
            {
                "source_episode_index": source_episode_index,
                "output_episode_index": output_episode_index,
                "source_frames": len(frame),
                "output_frames": len(episode["df"]),
                "pose_zero_frames_removed": zero_count,
                "frozen_pose_leadin_frames_removed": removed - zero_count,
            }
        )

    return cleaned_episodes, episode_reports


def filter_video_frames_streaming(
    video_path: Path,
    valid_indices: np.ndarray,
    fps: int,
) -> None:
    """Re-encode selected frames without loading an entire video into RAM."""

    selected = np.asarray(valid_indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError(f"valid video indices must be a non-empty vector: {video_path}")
    if selected[0] < 0 or np.any(selected[1:] <= selected[:-1]):
        raise ValueError(f"valid video indices must be sorted and unique: {video_path}")

    temporary_path = video_path.with_suffix(".tmp.mp4")
    if temporary_path.exists():
        raise FileExistsError(f"temporary video already exists: {temporary_path}")

    input_container = av.open(str(video_path))
    output_container: av.container.OutputContainer | None = None
    try:
        input_stream = input_container.streams.video[0]
        width = int(input_stream.codec_context.width)
        height = int(input_stream.codec_context.height)
        output_container = av.open(str(temporary_path), mode="w")
        output_stream = output_container.add_stream("h264", rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"

        selection_cursor = 0
        for frame_index, decoded_frame in enumerate(input_container.decode(input_stream)):
            if selection_cursor >= len(selected):
                break
            if frame_index != int(selected[selection_cursor]):
                continue
            rgb_frame = decoded_frame.to_ndarray(format="rgb24")
            output_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            selection_cursor += 1

        if selection_cursor != len(selected):
            raise ValueError(
                f"video {video_path} ended after selecting {selection_cursor}/"
                f"{len(selected)} requested frames"
            )
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        if output_container is not None:
            output_container.close()

    temporary_path.replace(video_path)


def rebuild_metadata_and_stats(dataset_path: Path) -> dict[str, int]:
    """Repair LeRobot v2.1 counts, indices, dtypes, and episode statistics."""

    from lerobot.common.datasets.compute_stats import (
        aggregate_stats,
        compute_episode_stats,
    )
    from lerobot.common.datasets.utils import (
        load_episodes_stats,
        serialize_dict,
        write_jsonlines,
    )

    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"

    info = load_info(dataset_path)
    episodes = read_jsonlines(episodes_path)
    tasks = read_jsonlines(tasks_path) if tasks_path.is_file() else []
    video_keys = get_video_keys(info)
    stats_features = {
        key: feature
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") not in {"video", "image"}
    }
    fps = int(info.get("fps", 50))
    chunk_size = int(info.get("chunks_size", 1000))

    total_frames = 0
    stats_records: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = get_parquet_path(dataset_path, info, episode_index)
        frame = pd.read_parquet(parquet_path)
        episode_length = len(frame)
        episode["length"] = episode_length

        frame["timestamp"] = (
            np.arange(episode_length, dtype=np.float32) / np.float32(fps)
        )
        frame["frame_index"] = np.arange(episode_length, dtype=np.int64)
        frame["episode_index"] = np.full(
            episode_length,
            episode_index,
            dtype=np.int64,
        )
        frame["index"] = np.arange(
            total_frames,
            total_frames + episode_length,
            dtype=np.int64,
        )
        frame.to_parquet(parquet_path)

        missing_features = set(stats_features) - set(frame.columns)
        if missing_features:
            raise ValueError(
                f"episode {episode_index} is missing stats columns: "
                f"{sorted(missing_features)}"
            )
        episode_data = {
            key: np.stack(frame[key].to_numpy())
            for key in stats_features
        }
        episode_stats = compute_episode_stats(episode_data, stats_features)
        stats_records.append(
            {
                "episode_index": episode_index,
                "stats": serialize_dict(episode_stats),
            }
        )
        total_frames += episode_length

    write_jsonlines(episodes, episodes_path)
    write_jsonlines(stats_records, stats_path)

    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info["total_videos"] = len(episodes) * len(video_keys)
    info["total_chunks"] = (
        (len(episodes) + chunk_size - 1) // chunk_size if episodes else 0
    )
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info.pop("discarded_episode_indices", None)
    info_path.write_text(
        json.dumps(info, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loaded_stats = load_episodes_stats(dataset_path)
    if len(loaded_stats) != len(episodes):
        raise ValueError(
            f"expected {len(episodes)} episode stats entries, got {len(loaded_stats)}"
        )
    aggregate_stats(list(loaded_stats.values()))
    return {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": len(episodes) * len(video_keys),
        "total_tasks": len(tasks),
    }


def inspect_video(video_path: Path) -> tuple[int, int, int, float]:
    """Decode an MP4 and return frame count, width, height, and frame rate."""

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        frame_rate = float(stream.average_rate) if stream.average_rate else 0.0
        frame_count = sum(1 for _frame in container.decode(stream))
        return frame_count, width, height, frame_rate
    finally:
        container.close()


def verify_dataset(
    dataset_path: Path,
    expected_episodes: int,
    expected_frames: int,
    pose_modes: Sequence[int],
) -> dict[str, int]:
    """Perform structural, temporal, video, stats, and loader checks."""

    from lerobot.common.datasets.compute_stats import compute_episode_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import load_episodes_stats

    info = load_info(dataset_path)
    episodes = read_jsonlines(dataset_path / "meta" / "episodes.jsonl")
    if info.get("discarded_episode_indices"):
        raise ValueError("discarded_episode_indices remains in cleaned info.json")
    if len(episodes) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, found {len(episodes)}"
        )
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    if episode_indices != list(range(len(episodes))):
        raise ValueError("cleaned episode indices are not contiguous")

    video_keys = get_video_keys(info)
    stats_features = {
        key: feature
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") not in {"video", "image"}
    }
    episode_stats = load_episodes_stats(dataset_path)
    if sorted(episode_stats) != list(range(len(episodes))):
        raise ValueError("episodes_stats indices are incomplete or not contiguous")
    fps = int(info.get("fps", 50))
    expected_global_index = 0
    total_frames = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        parquet_path = get_parquet_path(dataset_path, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing cleaned parquet: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        if len(frame) != episode_length:
            raise ValueError(f"episode length mismatch: {episode_index}")
        if frame["timestamp"].dtype != np.dtype("float32"):
            raise ValueError(f"timestamp is not float32: episode {episode_index}")

        expected_timestamp = (
            np.arange(episode_length, dtype=np.float32) / np.float32(fps)
        )
        if not np.array_equal(frame["timestamp"].to_numpy(), expected_timestamp):
            raise ValueError(f"timestamp sequence mismatch: episode {episode_index}")
        if not np.array_equal(
            frame["frame_index"].to_numpy(),
            np.arange(episode_length),
        ):
            raise ValueError(f"frame_index sequence mismatch: episode {episode_index}")
        if not np.all(frame["episode_index"].to_numpy() == episode_index):
            raise ValueError(f"episode_index column mismatch: episode {episode_index}")
        expected_indices = np.arange(
            expected_global_index,
            expected_global_index + episode_length,
        )
        if not np.array_equal(frame["index"].to_numpy(), expected_indices):
            raise ValueError(f"global index sequence mismatch: episode {episode_index}")

        smpl_pose = np.vstack(
            [np.asarray(value, dtype=np.float32) for value in frame[SMPL_POSE_COLUMN]]
        )
        stream_mode = frame[STREAM_MODE_COLUMN].to_numpy()
        remaining_pose_zero = np.all(smpl_pose == 0, axis=1) & np.isin(
            stream_mode,
            np.asarray(tuple(pose_modes)),
        )
        if np.any(remaining_pose_zero):
            raise ValueError(f"pose-mode zero SMPL remains: episode {episode_index}")

        stats_input = {
            key: np.stack(frame[key].to_numpy())
            for key in stats_features
        }
        recomputed_stats = compute_episode_stats(stats_input, stats_features)
        stored_stats = episode_stats[episode_index]
        if set(stored_stats) != set(recomputed_stats):
            raise ValueError(f"stats feature mismatch: episode {episode_index}")
        for feature_key, feature_stats in recomputed_stats.items():
            for statistic_name, expected_value in feature_stats.items():
                np.testing.assert_allclose(
                    stored_stats[feature_key][statistic_name],
                    expected_value,
                    rtol=1e-7,
                    atol=1e-7,
                    err_msg=(
                        f"stats mismatch: episode {episode_index}, "
                        f"feature {feature_key}, statistic {statistic_name}"
                    ),
                )

        for video_key in video_keys:
            video_path = get_video_path(
                dataset_path,
                info,
                episode_index,
                video_key,
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"missing cleaned video: {video_path}")
            frames_in_video, width, height, video_fps = inspect_video(video_path)
            if frames_in_video != episode_length:
                raise ValueError(
                    f"video length mismatch for {video_path}: "
                    f"{frames_in_video} != {episode_length}"
                )
            feature = info["features"][video_key]
            expected_height, expected_width = feature["shape"][:2]
            if (height, width) != (expected_height, expected_width):
                raise ValueError(
                    f"video resolution mismatch for {video_path}: "
                    f"{width}x{height} != {expected_width}x{expected_height}"
                )
            if not np.isclose(video_fps, fps):
                raise ValueError(
                    f"video fps mismatch for {video_path}: {video_fps} != {fps}"
                )

        total_frames += episode_length
        expected_global_index += episode_length

    if total_frames != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, found {total_frames}")
    expected_videos = len(episodes) * len(video_keys)
    expected_info = {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": expected_videos,
        "splits": {"train": f"0:{len(episodes)}"},
    }
    for key, expected_value in expected_info.items():
        if info.get(key) != expected_value:
            raise ValueError(
                f"info.json {key} mismatch: {info.get(key)!r} != {expected_value!r}"
            )

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        for feature_stats in episode_stats[episode_index].values():
            if int(feature_stats["count"][0]) != episode_length:
                raise ValueError(f"stats count mismatch: episode {episode_index}")

    metadata = LeRobotDatasetMetadata(
        f"local/{dataset_path.name}",
        root=dataset_path,
    )
    if metadata.total_episodes != len(episodes) or metadata.total_frames != total_frames:
        raise ValueError("LeRobot metadata loader count mismatch")

    parquet_count = sum(1 for _path in dataset_path.rglob("*.parquet"))
    video_count = sum(1 for _path in dataset_path.rglob("*.mp4"))
    temporary_video_count = sum(1 for _path in dataset_path.rglob("*.tmp.mp4"))
    if parquet_count != len(episodes):
        raise ValueError(f"unexpected parquet count: {parquet_count}")
    if video_count != expected_videos:
        raise ValueError(f"unexpected video count: {video_count}")
    if temporary_video_count:
        raise ValueError(f"found {temporary_video_count} temporary videos")

    return {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": expected_videos,
        "stats_features": len(metadata.stats),
    }


def unique_backup_path(output_path: Path) -> Path:
    """Choose a non-conflicting timestamped backup name."""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base = output_path.with_name(f"{output_path.name}_backup_{timestamp}")
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = output_path.with_name(f"{base.name}_{counter}")
        counter += 1
    return candidate


def promote_output(
    partial_path: Path,
    output_path: Path,
    replace_existing: bool,
) -> Path | None:
    """Atomically promote a verified partial output, backing up old output."""

    backup_path: Path | None = None
    if output_path.exists():
        if not replace_existing:
            raise FileExistsError(
                f"output already exists: {output_path}; use --replace-existing "
                "to rotate it to a backup"
            )
        backup_path = unique_backup_path(output_path)
        output_path.rename(backup_path)

    try:
        partial_path.rename(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        raise
    return backup_path


def process_job(
    job: DatasetJob,
    scan: DatasetScan,
    processor: ModuleType,
    pose_modes: Sequence[int],
    extra_discarded: Sequence[int],
    replace_existing: bool,
) -> tuple[dict[str, int], Path | None]:
    """Build, verify, and promote one cleaned dataset."""

    source_hashes_before = metadata_hashes(job.source)
    LOGGER.info("Reading discard-marked episodes from %s", job.source)
    _stats, processed_episodes, reference_info = processor.process_single_dataset(
        job.source,
        remove_stale_smpl=False,
        remove_discarded=True,
    )
    if extra_discarded:
        LOGGER.info(
            "Removing explicitly discarded source episodes: %s",
            sorted(set(int(index) for index in extra_discarded)),
        )
        processed_episodes = remove_extra_discarded_episodes(
            processed_episodes,
            extra_discarded,
        )
    validate_processed_episode_indices(
        job.source,
        processed_episodes,
        extra_discarded,
    )
    cleaned_episodes, episode_reports = apply_pose_cleanup(
        processed_episodes,
        pose_modes,
    )
    if not cleaned_episodes:
        raise ValueError(f"no episodes remain after cleaning {job.source}")

    tasks = processor.load_tasks_meta(job.source)
    script_config = reference_info.get("script_config")
    processor.filter_video_frames = filter_video_frames_streaming
    LOGGER.info("Writing partial cleaned dataset to %s", job.partial)
    processor.write_output_dataset(
        job.partial,
        cleaned_episodes,
        reference_info,
        tasks,
        script_config,
    )
    processor.copy_modality_json([job.source], job.partial)

    rebuilt = rebuild_metadata_and_stats(job.partial)
    report = {
        "policy": {
            "remove_discarded_episodes": True,
            "extra_discarded_episode_indices": sorted(
                set(int(index) for index in extra_discarded)
            ),
            "remove_zero_smpl_only_in_stream_modes": list(pose_modes),
            "remove_frozen_pose_leadin": True,
            "preserve_planner_and_off_zero_smpl": True,
        },
        "scan": asdict(scan),
        "result": rebuilt,
        "episode_mapping": episode_reports,
        "final_output": str(job.output),
    }
    report_path = job.partial / "meta" / "cleanup_report.json"
    report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Verifying parquet, video, statistics, and LeRobot metadata")
    verified = verify_dataset(
        job.partial,
        expected_episodes=scan.expected_output_episodes,
        expected_frames=scan.expected_output_frames,
        pose_modes=pose_modes,
    )
    source_hashes_after = metadata_hashes(job.source)
    if source_hashes_before != source_hashes_after:
        raise RuntimeError(f"source metadata changed unexpectedly: {job.source}")

    backup_path = promote_output(job.partial, job.output, replace_existing)
    return verified, backup_path


def build_jobs(
    dataset_paths: Sequence[Path],
    output_suffix: str,
    replace_existing: bool,
) -> list[DatasetJob]:
    """Resolve and validate all jobs before any output is written."""

    if not output_suffix:
        raise ValueError("--output-suffix cannot be empty")
    jobs: list[DatasetJob] = []
    seen_outputs: set[Path] = set()
    sources = [path.expanduser().resolve() for path in dataset_paths]
    for raw_path in dataset_paths:
        source = raw_path.expanduser().resolve()
        validate_source(source)
        output = source.with_name(f"{source.name}{output_suffix}")
        partial = output.with_name(f".{output.name}.partial")
        if any(
            output == path or output in path.parents or path in output.parents
            for path in sources
        ):
            raise ValueError(f"output would replace an input dataset: {output}")
        if output.is_symlink() or partial.is_symlink():
            raise ValueError(f"output paths must not be symlinks: {output}")
        if output in seen_outputs:
            raise ValueError(f"duplicate output path: {output}")
        if output.exists() and not replace_existing:
            raise FileExistsError(
                f"output already exists: {output}; pass --replace-existing to "
                "rotate it to a timestamped backup after the new copy verifies"
            )
        if partial.exists():
            raise FileExistsError(
                f"partial output already exists from an earlier run: {partial}"
            )
        seen_outputs.add(output)
        jobs.append(DatasetJob(source=source, output=output, partial=partial))
    return jobs


def print_scan(scan: DatasetScan) -> None:
    """Print a concise human-readable dry-run summary."""

    print(f"\nDataset: {scan.source}")
    print(f"  Output:                         {scan.output}")
    print(
        f"  Discard episodes:               {scan.discarded_episodes} "
        f"({scan.discarded_frames} frames)"
    )
    print(
        f"  Extra discard episodes:         {scan.extra_discarded_episodes} "
        f"({scan.extra_discarded_frames} frames)"
    )
    print(f"  Zero SMPL by stream mode:       {scan.zero_smpl_by_stream_mode}")
    print(f"  POSE/POSE_PAUSE zero frames:    {scan.pose_zero_frames}")
    print(f"  Frozen pose lead-in frames:     {scan.frozen_pose_leadin_frames}")
    print(f"  Mode-aware frames removed:      {scan.pose_frames_to_remove}")
    print(f"  Expected cleaned episodes:      {scan.expected_output_episodes}")
    print(f"  Expected cleaned frames:        {scan.expected_output_frames}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=Path,
        help="One or more source LeRobot dataset directories.",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help=f"Suffix for sibling output directories (default: {DEFAULT_OUTPUT_SUFFIX}).",
    )
    parser.add_argument(
        "--processor-path",
        type=Path,
        default=os.environ.get("SONIC_DATASET_PROCESSOR"),
        help=(
            "Path to GEAR-SONIC gear_sonic/scripts/process_dataset.py "
            "(or set SONIC_DATASET_PROCESSOR); required except for --dry-run."
        ),
    )
    parser.add_argument(
        "--pose-modes",
        nargs="+",
        type=int,
        default=list(DEFAULT_POSE_MODES),
        help="Stream modes where zero SMPL means stale data (default: 1 4).",
    )
    parser.add_argument(
        "--extra-discard-episode",
        action="append",
        type=int,
        default=[],
        help=(
            "Additional source episode index to exclude without modifying "
            "source metadata; repeat for multiple indices and use with exactly "
            "one input dataset."
        ),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="After verification, rotate an existing output to a timestamped backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect and report expected changes; do not write data.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and args.processor_path is None:
        parser.error("--processor-path or SONIC_DATASET_PROCESSOR is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    if not args.output_suffix:
        raise ValueError("--output-suffix cannot be empty")
    if args.extra_discard_episode and len(args.dataset_paths) != 1:
        raise ValueError(
            "--extra-discard-episode requires exactly one input dataset"
        )

    jobs = build_jobs(
        args.dataset_paths,
        output_suffix=args.output_suffix,
        replace_existing=args.replace_existing,
    )
    scans = [
        scan_dataset(job, args.pose_modes, args.extra_discard_episode)
        for job in jobs
    ]
    for scan in scans:
        print_scan(scan)

    if args.dry_run:
        print("\nDry run complete; no files were changed.")
        return 0

    processor = load_processor(args.processor_path.expanduser().resolve())
    for job, scan in zip(jobs, scans):
        LOGGER.info("Starting mode-aware cleanup: %s", job.source)
        try:
            verified, backup_path = process_job(
                job,
                scan,
                processor,
                args.pose_modes,
                args.extra_discard_episode,
                args.replace_existing,
            )
        except Exception:
            LOGGER.exception(
                "Cleanup failed. Source is unchanged; partial output is kept at %s",
                job.partial,
            )
            return 1

        LOGGER.info("Cleanup verified and promoted to %s", job.output)
        if backup_path is not None:
            LOGGER.info("Previous output preserved at %s", backup_path)
        LOGGER.info("Verified result: %s", verified)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
