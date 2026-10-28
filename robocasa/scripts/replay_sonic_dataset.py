#!/usr/bin/env python3
"""Replay and validate a collected SONIC LeRobot dataset.

The SONIC VLA exporter stores one parquet file and one video per camera for each
episode.  It does not store the complete MuJoCo state needed for simulator
playback.  This tool therefore provides a synchronized, offline replay of the
recorded camera streams together with structural and temporal validation of the
LeRobot data.

Each replay frame contains the selected camera views plus episode, task, time,
and action/state tracking information.  A JSON report records every check so a
reviewer can distinguish a structurally reliable export from a task-success
decision that still requires visual inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
import json
import logging
import math
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("replay_sonic_dataset")
REPORT_SCHEMA_VERSION = 1
DEFAULT_NUM_EPISODES = 5
DEFAULT_SEED = 0
DEFAULT_MAX_OUTPUT_WIDTH = 1920
POSE_STREAM_MODES = (1, 4)  # POSE, POSE_PAUSE


@dataclass
class VideoCheck:
    """Validation result for one encoded camera stream."""

    key: str
    path: str
    expected_frames: int
    frames: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duplicate_frame_ratio: float | None = None
    reliable: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class FeatureCheck:
    """Shape and finite-value checks for one parquet feature."""

    expected_shape: list[int]
    rows: int
    invalid_shape_rows: int = 0
    non_finite_values: int = 0
    reliable: bool = True


@dataclass
class EpisodeCheck:
    """Complete validation and replay result for one episode."""

    episode_index: int
    tasks: list[str]
    expected_frames: int
    parquet_path: str
    discarded: bool
    reliable: bool = False
    semantic_success: str = "not_available"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feature_checks: dict[str, FeatureCheck] = field(default_factory=dict)
    videos: dict[str, VideoCheck] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    replay_video: str | None = None


def read_jsonlines(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset_metadata(
    dataset_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and minimally validate LeRobot dataset metadata."""

    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_path}")
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_jsonlines(episodes_path)
    if not episodes:
        raise ValueError(f"dataset has no episodes: {dataset_path}")

    indices = [int(episode["episode_index"]) for episode in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("meta/episodes.jsonl contains duplicate episode indices")
    return info, episodes


def get_video_keys(info: dict[str, Any]) -> list[str]:
    """Return video feature keys in metadata order."""

    explicit = info.get("video_keys")
    if explicit:
        return list(explicit)
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def get_parquet_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    """Resolve one episode parquet path using LeRobot metadata templates."""

    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_path / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def get_video_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    """Resolve one episode video path using LeRobot metadata templates."""

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


def resolve_camera_keys(
    requested: Sequence[str] | None,
    available: Sequence[str],
) -> list[str]:
    """Resolve full feature keys while accepting short names such as ``ego_view``."""

    if not available:
        raise ValueError("dataset metadata contains no video features")
    if requested is None:
        return list(available)

    resolved: list[str] = []
    for candidate in requested:
        if candidate in available:
            match = candidate
        else:
            matches = [
                key for key in available if key.rsplit(".", maxsplit=1)[-1] == candidate
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"unknown or ambiguous camera key {candidate!r}; "
                    f"available keys: {', '.join(available)}"
                )
            match = matches[0]
        if match not in resolved:
            resolved.append(match)
    return resolved


def select_episodes(
    episode_records: Sequence[dict[str, Any]],
    discarded_indices: set[int],
    requested: Sequence[int] | None,
    num_episodes: int | None,
    include_all: bool,
    include_discarded: bool,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Select episodes deterministically, excluding discarded recordings by default."""

    by_index = {int(record["episode_index"]): record for record in episode_records}
    if requested is not None:
        requested_indices = [int(index) for index in requested]
        if len(requested_indices) != len(set(requested_indices)):
            raise ValueError("--episodes contains duplicate indices")
        unknown = sorted(set(requested_indices) - set(by_index))
        if unknown:
            raise ValueError(
                f"episode indices are not present in the dataset: {unknown}"
            )
        requested_discarded = sorted(set(requested_indices) & discarded_indices)
        if requested_discarded and not include_discarded:
            raise ValueError(
                "requested episodes are marked discarded: "
                f"{requested_discarded}; pass --include-discarded to inspect them"
            )
        selected = [by_index[index] for index in requested_indices]
    else:
        selected = [
            record
            for record in episode_records
            if include_discarded
            or int(record["episode_index"]) not in discarded_indices
        ]
        selected.sort(key=lambda record: int(record["episode_index"]))
        if shuffle:
            random.Random(seed).shuffle(selected)
        if not include_all and num_episodes is not None:
            selected = selected[:num_episodes]

    if not include_discarded:
        selected = [
            record
            for record in selected
            if int(record["episode_index"]) not in discarded_indices
        ]
    if not selected:
        raise ValueError("episode selection is empty")
    return selected


def _expected_feature_size(feature: dict[str, Any]) -> tuple[list[int], int]:
    shape = feature.get("shape", [1])
    if isinstance(shape, int):
        shape = [shape]
    expected_shape = [int(value) for value in shape]
    expected_size = math.prod(expected_shape) if expected_shape else 1
    return expected_shape, expected_size


def validate_feature(
    values: pd.Series,
    feature: dict[str, Any],
) -> tuple[FeatureCheck, np.ndarray | None]:
    """Check row shape and finite values, returning a stack when shapes are valid."""

    expected_shape, expected_size = _expected_feature_size(feature)
    arrays: list[np.ndarray] = []
    invalid_shape_rows = 0
    non_finite_values = 0
    numeric = feature.get("dtype") not in {"string", "video", "image"}

    for value in values:
        array = np.asarray(value)
        if array.size != expected_size:
            invalid_shape_rows += 1
            continue
        flat = array.reshape(-1)
        if numeric:
            try:
                numeric_values = flat.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                invalid_shape_rows += 1
                continue
            non_finite_values += int((~np.isfinite(numeric_values)).sum())
            arrays.append(numeric_values)
        else:
            arrays.append(flat)

    check = FeatureCheck(
        expected_shape=expected_shape,
        rows=len(values),
        invalid_shape_rows=invalid_shape_rows,
        non_finite_values=non_finite_values,
        reliable=invalid_shape_rows == 0 and non_finite_values == 0,
    )
    stacked = None
    if check.reliable and arrays and numeric:
        stacked = np.stack(arrays)
    return check, stacked


def _norm_summary(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "max": float(np.max(norms)),
    }


def validate_parquet(
    parquet_path: Path,
    info: dict[str, Any],
    episode_index: int,
    expected_frames: int,
    fps: float,
) -> tuple[pd.DataFrame | None, dict[str, FeatureCheck], dict[str, Any], list[str]]:
    """Validate one parquet file and summarize action/state tracking."""

    errors: list[str] = []
    feature_checks: dict[str, FeatureCheck] = {}
    telemetry: dict[str, Any] = {}
    if not parquet_path.is_file():
        return None, feature_checks, telemetry, [f"missing parquet: {parquet_path}"]

    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as error:  # pragma: no cover - backend-specific detail
        return (
            None,
            feature_checks,
            telemetry,
            [f"failed to read parquet {parquet_path}: {error}"],
        )

    if len(frame) != expected_frames:
        errors.append(
            f"parquet has {len(frame)} rows; metadata declares {expected_frames}"
        )

    if "frame_index" not in frame:
        errors.append("parquet is missing frame_index")
    elif not np.array_equal(
        frame["frame_index"].to_numpy(), np.arange(len(frame), dtype=np.int64)
    ):
        errors.append("frame_index is not a contiguous zero-based sequence")

    if "episode_index" not in frame:
        errors.append("parquet is missing episode_index")
    elif not np.all(frame["episode_index"].to_numpy() == episode_index):
        errors.append("episode_index column does not match episode metadata")

    if "timestamp" not in frame:
        errors.append("parquet is missing timestamp")
    else:
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        expected_timestamps = (
            np.arange(len(frame), dtype=np.float32) / np.float32(fps)
        ).astype(np.float64)
        timestamp_tolerance = max(1e-6, 1e-5 / fps)
        if not np.all(np.isfinite(timestamps)):
            errors.append("timestamp contains NaN or Inf")
        elif not np.allclose(
            timestamps,
            expected_timestamps,
            rtol=0.0,
            atol=timestamp_tolerance,
        ):
            errors.append(f"timestamp is not frame_index / {fps:g} Hz")

    if "index" in frame and len(frame) > 1:
        global_indices = frame["index"].to_numpy(dtype=np.int64)
        if not np.all(np.diff(global_indices) == 1):
            errors.append("global index is not contiguous within the episode")

    arrays: dict[str, np.ndarray] = {}
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") in {"video", "image"}:
            continue
        if key not in frame:
            errors.append(f"parquet is missing feature {key!r}")
            continue
        check, stacked = validate_feature(frame[key], feature)
        feature_checks[key] = check
        if not check.reliable:
            errors.append(
                f"feature {key!r} has {check.invalid_shape_rows} invalid-shape "
                f"rows and {check.non_finite_values} non-finite values"
            )
        if stacked is not None:
            arrays[key] = stacked

    for key in ("action.wbc", "observation.state", "observation.eef_state"):
        if key in arrays:
            telemetry[f"{key}.norm"] = _norm_summary(arrays[key])

    action = arrays.get("action.wbc")
    state = arrays.get("observation.state")
    if action is not None and state is not None and action.shape == state.shape:
        tracking_error = np.abs(action - state)
        telemetry["action_state_tracking"] = {
            "mean_absolute_error": float(np.mean(tracking_error)),
            "p95_absolute_error": float(np.quantile(tracking_error, 0.95)),
            "max_absolute_error": float(np.max(tracking_error)),
            "mean_l2_error": float(np.mean(np.linalg.norm(action - state, axis=1))),
        }

    smpl_pose = arrays.get("teleop.smpl_pose")
    stream_mode = arrays.get("teleop.stream_mode")
    if smpl_pose is not None and stream_mode is not None:
        modes = stream_mode.reshape(-1)
        pose_mode_zero = np.all(smpl_pose == 0, axis=1) & np.isin(
            modes, np.asarray(POSE_STREAM_MODES)
        )
        stale_count = int(pose_mode_zero.sum())
        telemetry["pose_mode_zero_smpl_frames"] = stale_count
        if stale_count:
            errors.append(
                f"teleop.smpl_pose is zero in a pose-driven mode for {stale_count} frames"
            )

    return frame, feature_checks, telemetry, errors


def inspect_video(
    video_path: Path,
    key: str,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
) -> VideoCheck:
    """Decode a video and compare its length, resolution, and FPS to metadata."""

    check = VideoCheck(
        key=key,
        path=str(video_path),
        expected_frames=expected_frames,
    )
    if not video_path.is_file():
        check.errors.append(f"missing video: {video_path}")
        return check

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        check.errors.append(f"unable to open video: {video_path}")
        capture.release()
        return check

    try:
        check.width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        check.height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        check.fps = float(capture.get(cv2.CAP_PROP_FPS))
        duplicate_frames = 0
        previous_frame: np.ndarray | None = None
        while True:
            ok, decoded = capture.read()
            if not ok:
                break
            if previous_frame is not None:
                difference = cv2.absdiff(decoded, previous_frame)
                if float(np.mean(difference)) <= 0.5:
                    duplicate_frames += 1
            previous_frame = decoded
            check.frames += 1
    finally:
        capture.release()

    if check.frames > 1:
        check.duplicate_frame_ratio = duplicate_frames / (check.frames - 1)
    if check.frames != expected_frames:
        check.errors.append(
            f"video has {check.frames} frames; expected {expected_frames}"
        )
    if (check.width, check.height) != (expected_width, expected_height):
        check.errors.append(
            f"video resolution is {check.width}x{check.height}; "
            f"expected {expected_width}x{expected_height}"
        )
    if not np.isclose(check.fps, expected_fps, rtol=0.0, atol=0.01):
        check.errors.append(f"video FPS is {check.fps:g}; expected {expected_fps:g}")
    check.reliable = not check.errors
    return check


def _camera_label(key: str) -> str:
    return key.rsplit(".", maxsplit=1)[-1]


def _fit_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    """Trim an OpenCV text label to fit a target width."""

    if (
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        <= max_width
    ):
        return text
    suffix = "..."
    while (
        text
        and cv2.getTextSize(text + suffix, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,)[
            0
        ][0]
        > max_width
    ):
        text = text[:-1]
    return text + suffix


def render_episode_replay(
    output_path: Path,
    video_checks: dict[str, VideoCheck],
    camera_keys: Sequence[str],
    frame: pd.DataFrame,
    episode_index: int,
    tasks: Sequence[str],
    source_fps: float,
    video_stride: int,
    max_output_width: int,
    overlay: bool,
) -> int:
    """Stream synchronized camera frames into a tiled replay MP4."""

    aspect_sum = sum(
        video_checks[key].width / video_checks[key].height for key in camera_keys
    )
    native_height = min(video_checks[key].height for key in camera_keys)
    tile_height = min(native_height, int(max_output_width / aspect_sum))
    tile_height = max(2, tile_height - tile_height % 2)
    tile_widths = [
        max(
            2,
            int(round(video_checks[key].width * tile_height / video_checks[key].height))
            // 2
            * 2,
        )
        for key in camera_keys
    ]
    canvas_width = sum(tile_widths)
    banner_height = 0
    if overlay:
        banner_height = max(72, min(96, tile_height // 4))
        banner_height += banner_height % 2
    canvas_height = tile_height + banner_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = source_fps / video_stride
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"unable to create replay video: {output_path}")

    captures = {key: cv2.VideoCapture(video_checks[key].path) for key in camera_keys}
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        writer.release()
        raise RuntimeError("unable to reopen one or more camera videos for replay")

    task = tasks[0] if tasks else "task unavailable"
    written_frames = 0
    expected_frames = len(frame)
    try:
        for frame_index in range(expected_frames):
            decoded: dict[str, np.ndarray] = {}
            for key, capture in captures.items():
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"camera {key!r} ended before frame {frame_index}"
                    )
                decoded[key] = image

            if frame_index % video_stride != 0 and frame_index != expected_frames - 1:
                continue

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            x = 0
            for key, width in zip(camera_keys, tile_widths):
                tile = cv2.resize(
                    decoded[key],
                    (width, tile_height),
                    interpolation=cv2.INTER_AREA,
                )
                canvas[banner_height:, x : x + width] = tile
                if overlay:
                    cv2.rectangle(
                        canvas,
                        (x + 8, banner_height + 8),
                        (x + 220, banner_height + 38),
                        (0, 0, 0),
                        thickness=-1,
                    )
                    cv2.putText(
                        canvas,
                        _camera_label(key),
                        (x + 16, banner_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                x += width

            if overlay:
                timestamp = float(frame["timestamp"].iloc[frame_index])
                action_norm = float(
                    np.linalg.norm(np.asarray(frame["action.wbc"].iloc[frame_index]))
                )
                state_norm = float(
                    np.linalg.norm(
                        np.asarray(frame["observation.state"].iloc[frame_index])
                    )
                )
                tracking_mae = float(
                    np.mean(
                        np.abs(
                            np.asarray(frame["action.wbc"].iloc[frame_index])
                            - np.asarray(frame["observation.state"].iloc[frame_index])
                        )
                    )
                )
                top_line = (
                    f"episode {episode_index:06d} | frame {frame_index + 1}/{expected_frames} "
                    f"| t={timestamp:.2f}s | ||action.wbc||={action_norm:.3f} "
                    f"| ||state||={state_norm:.3f} | mean|action-state|={tracking_mae:.3f}"
                )
                text_scale = 0.56 if canvas_width >= 1280 else 0.42
                top_line = _fit_text(top_line, canvas_width - 24, text_scale, 1)
                task_line = _fit_text(f"task: {task}", canvas_width - 24, text_scale, 1)
                cv2.putText(
                    canvas,
                    top_line,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    task_line,
                    (12, min(banner_height - 12, 58)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
            written_frames += 1
    finally:
        for capture in captures.values():
            capture.release()
        writer.release()
    return written_frames


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def default_output_dir(dataset_path: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "artifacts" / "dataset_replay" / dataset_path.name / timestamp


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Validate selected episodes, write replays, and return the report."""

    dataset_path = args.dataset.expanduser().resolve()
    info, episode_records = load_dataset_metadata(dataset_path)
    fps = float(info.get("fps", 0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps!r}")

    available_camera_keys = get_video_keys(info)
    camera_keys = resolve_camera_keys(args.camera_keys, available_camera_keys)
    discarded_indices = {
        int(index) for index in info.get("discarded_episode_indices", [])
    }
    known_episode_indices = {int(record["episode_index"]) for record in episode_records}
    dataset_errors: list[str] = []
    metadata_episode_count = int(info.get("total_episodes", len(episode_records)))
    metadata_frame_count = int(
        info.get(
            "total_frames",
            sum(int(record["length"]) for record in episode_records),
        )
    )
    actual_metadata_frames = sum(int(record["length"]) for record in episode_records)
    if metadata_episode_count != len(episode_records):
        dataset_errors.append(
            f"info.json declares {metadata_episode_count} episodes; "
            f"episodes.jsonl contains {len(episode_records)}"
        )
    if metadata_frame_count != actual_metadata_frames:
        dataset_errors.append(
            f"info.json declares {metadata_frame_count} frames; "
            f"episode lengths sum to {actual_metadata_frames}"
        )
    expected_video_count = len(episode_records) * len(available_camera_keys)
    metadata_video_count = int(info.get("total_videos", expected_video_count))
    if metadata_video_count != expected_video_count:
        dataset_errors.append(
            f"info.json declares {metadata_video_count} videos; "
            f"episode/video-key counts imply {expected_video_count}"
        )
    unknown_discarded = sorted(discarded_indices - known_episode_indices)
    if unknown_discarded:
        dataset_errors.append(
            f"discarded_episode_indices contains unknown indices: {unknown_discarded}"
        )
    selected = select_episodes(
        episode_records=episode_records,
        discarded_indices=discarded_indices,
        requested=args.episodes,
        num_episodes=args.num_episodes,
        include_all=args.all,
        include_discarded=args.include_discarded,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(dataset_path)
    )
    if output_dir == dataset_path or dataset_path in output_dir.parents:
        raise ValueError(f"output must be outside the source dataset: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse replay output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_checks: list[EpisodeCheck] = []
    features = info.get("features", {})
    for position, episode_record in enumerate(selected, start=1):
        episode_index = int(episode_record["episode_index"])
        expected_frames = int(episode_record["length"])
        tasks = [str(task) for task in episode_record.get("tasks", [])]
        parquet_path = get_parquet_path(dataset_path, info, episode_index)
        check = EpisodeCheck(
            episode_index=episode_index,
            tasks=tasks,
            expected_frames=expected_frames,
            parquet_path=str(parquet_path),
            discarded=episode_index in discarded_indices,
        )
        if check.discarded:
            check.errors.append("episode is marked discarded in meta/info.json")
        if not tasks:
            check.warnings.append("episode metadata contains no task description")

        LOGGER.info(
            "[%d/%d] validating episode %06d (%d frames)",
            position,
            len(selected),
            episode_index,
            expected_frames,
        )
        frame, feature_checks, telemetry, parquet_errors = validate_parquet(
            parquet_path=parquet_path,
            info=info,
            episode_index=episode_index,
            expected_frames=expected_frames,
            fps=fps,
        )
        check.feature_checks = feature_checks
        check.telemetry = telemetry
        check.errors.extend(parquet_errors)

        for video_key in available_camera_keys:
            feature = features[video_key]
            shape, _ = _expected_feature_size(feature)
            if len(shape) < 2:
                video_check = VideoCheck(
                    key=video_key,
                    path=str(
                        get_video_path(dataset_path, info, episode_index, video_key)
                    ),
                    expected_frames=expected_frames,
                    errors=[f"video feature {video_key!r} has invalid shape {shape}"],
                )
            else:
                video_check = inspect_video(
                    video_path=get_video_path(
                        dataset_path, info, episode_index, video_key
                    ),
                    key=video_key,
                    expected_frames=expected_frames,
                    expected_width=int(shape[1]),
                    expected_height=int(shape[0]),
                    expected_fps=fps,
                )
            check.videos[video_key] = video_check
            check.errors.extend(video_check.errors)

        selected_video_checks_pass = all(
            check.videos[key].reliable for key in camera_keys
        )
        if not args.check_only and frame is not None and selected_video_checks_pass:
            replay_path = output_dir / f"episode_{episode_index:06d}_replay.mp4"
            try:
                written_frames = render_episode_replay(
                    output_path=replay_path,
                    video_checks=check.videos,
                    camera_keys=camera_keys,
                    frame=frame,
                    episode_index=episode_index,
                    tasks=tasks,
                    source_fps=fps,
                    video_stride=args.video_stride,
                    max_output_width=args.max_output_width,
                    overlay=not args.no_overlay,
                )
                check.replay_video = str(replay_path)
                check.telemetry["replay_output_frames"] = written_frames
            except Exception as error:  # pragma: no cover - codec-specific detail
                check.errors.append(f"failed to render replay video: {error}")
        elif not args.check_only and not selected_video_checks_pass:
            check.errors.append(
                "replay skipped because a selected camera failed validation"
            )

        check.reliable = not check.errors
        episode_checks.append(check)

    reliable_count = sum(check.reliable for check in episode_checks)
    passed = reliable_count == len(episode_checks) and not dataset_errors
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "name": dataset_path.name,
            "codebase_version": info.get("codebase_version"),
            "fps": fps,
            "total_episodes": int(info.get("total_episodes", len(episode_records))),
            "total_frames": int(info.get("total_frames", 0)),
            "discarded_episode_indices": sorted(discarded_indices),
            "video_keys": available_camera_keys,
            "errors": dataset_errors,
        },
        "selection": {
            "episode_indices": [check.episode_index for check in episode_checks],
            "camera_keys": camera_keys,
            "include_discarded": args.include_discarded,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "video_stride": args.video_stride,
        },
        "runtime": {
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv_python": _package_version("opencv-python"),
            "platform": platform.platform(),
        },
        "capabilities": {
            "synchronized_camera_replay": True,
            "parquet_video_alignment_check": True,
            "mujoco_state_replay": False,
            "automatic_task_success": False,
            "note": (
                "This LeRobot export has no full MuJoCo state/model sidecar. "
                "Use the collector demo.hdf5 for simulator state or action replay."
            ),
        },
        "summary": {
            "selected_episodes": len(episode_checks),
            "reliable_episodes": reliable_count,
            "failed_episodes": len(episode_checks) - reliable_count,
            "dataset_errors": len(dataset_errors),
            "passed": passed,
        },
        "episodes": [asdict(check) for check in episode_checks],
    }
    report_path = output_dir / "replay_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report, report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="SONIC LeRobot dataset directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--episodes",
        "--episode-index",
        type=int,
        nargs="+",
        default=None,
        help="specific episode indices, in replay order",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="validate and replay every eligible episode",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"number of episodes when --episodes/--all is omitted (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle eligible episodes before applying --num-episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed used by --shuffle (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="allow episodes listed in discarded_episode_indices to be selected",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help="camera feature keys or short names; defaults to all video features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="replay/report directory (default: artifacts/dataset_replay/<dataset>/<timestamp>)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="write the JSON report without encoding replay videos",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="write every Nth frame while preserving real-time duration (default: 1)",
    )
    parser.add_argument(
        "--max-output-width",
        type=int,
        default=DEFAULT_MAX_OUTPUT_WIDTH,
        help=f"maximum tiled replay width (default: {DEFAULT_MAX_OUTPUT_WIDTH})",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="omit task, timestamp, action/state norms, and camera labels",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.video_stride <= 0:
        parser.error("--video-stride must be positive")
    if args.max_output_width < 2:
        parser.error("--max-output-width must be at least 2")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        report, report_path = run_replay(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 2

    summary = report["summary"]
    LOGGER.info("reliability report: %s", report_path)
    LOGGER.info(
        "result: %d/%d selected episodes passed structural replay checks",
        summary["reliable_episodes"],
        summary["selected_episodes"],
    )
    if not report["capabilities"]["mujoco_state_replay"]:
        LOGGER.info(
            "task success remains a visual decision; simulator replay requires demo.hdf5"
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
