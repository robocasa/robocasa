from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np
import pandas as pd
import pytest


FPS = 10.0
FRAME_COUNT = 4
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
CAMERA_KEYS = (
    "observation.images.ego_view",
    "observation.images.left_wrist",
)
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "scripts"
    / "replay_sonic_dataset.py"
)


def _load_replay_module() -> ModuleType:
    """Load the standalone CLI without importing robocasa and robosuite."""

    module_name = "replay_sonic_dataset_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_replay_module()


def _video_path(dataset_path: Path, episode_index: int, camera_key: str) -> Path:
    return (
        dataset_path
        / "videos"
        / "chunk-000"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )


def _parquet_path(dataset_path: Path, episode_index: int) -> Path:
    return dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _write_video(
    path: Path,
    *,
    frame_count: int,
    color: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened(), f"OpenCV could not create test video {path}"
    try:
        for frame_index in range(frame_count):
            frame = np.full(
                (FRAME_HEIGHT, FRAME_WIDTH, 3),
                color,
                dtype=np.uint8,
            )
            stripe_start = frame_index * 4
            frame[:, stripe_start : stripe_start + 2] = (255, 255, 255)
            writer.write(frame)
    finally:
        writer.release()


def _write_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "sonic_lerobot"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 2 * FRAME_COUNT,
        "total_videos": 2 * len(CAMERA_KEYS),
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/" "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "discarded_episode_indices": [1],
        "features": {
            CAMERA_KEYS[0]: {
                "dtype": "video",
                "shape": [FRAME_HEIGHT, FRAME_WIDTH, 3],
            },
            CAMERA_KEYS[1]: {
                "dtype": "video",
                "shape": [FRAME_HEIGHT, FRAME_WIDTH, 3],
            },
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action.wbc": {"dtype": "float32", "shape": [2]},
        },
    }
    (meta_path / "info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    episodes = [
        {
            "episode_index": episode_index,
            "tasks": ["Test the synchronized replay."],
            "length": FRAME_COUNT,
        }
        for episode_index in range(2)
    ]
    (meta_path / "episodes.jsonl").write_text(
        "".join(f"{json.dumps(episode)}\n" for episode in episodes),
        encoding="utf-8",
    )

    camera_colors = {
        CAMERA_KEYS[0]: (10, 20, 220),
        CAMERA_KEYS[1]: (10, 220, 20),
    }
    for episode_index in range(2):
        parquet_path = _parquet_path(dataset_path, episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        global_start = episode_index * FRAME_COUNT
        pd.DataFrame(
            {
                "observation.state": [
                    [float(frame_index), 0.5] for frame_index in range(FRAME_COUNT)
                ],
                "action.wbc": [
                    [float(frame_index) + 0.1, 0.4]
                    for frame_index in range(FRAME_COUNT)
                ],
                "timestamp": np.arange(FRAME_COUNT, dtype=np.float32) / FPS,
                "frame_index": np.arange(FRAME_COUNT, dtype=np.int64),
                "episode_index": np.full(
                    FRAME_COUNT,
                    episode_index,
                    dtype=np.int64,
                ),
                "index": np.arange(
                    global_start,
                    global_start + FRAME_COUNT,
                    dtype=np.int64,
                ),
            }
        ).to_parquet(parquet_path, index=False)

        for camera_key, color in camera_colors.items():
            _write_video(
                _video_path(dataset_path, episode_index, camera_key),
                frame_count=FRAME_COUNT,
                color=color,
            )

    return dataset_path


def _read_report(output_dir: Path) -> dict:
    return json.loads((output_dir / "replay_report.json").read_text(encoding="utf-8"))


def _run(
    dataset_path: Path,
    output_dir: Path,
    *extra_args: str,
) -> int:
    return replay.main(
        [
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "ERROR",
            *extra_args,
        ]
    )


def test_replay_refuses_output_inside_source(tmp_path):
    dataset = _write_dataset(tmp_path)
    output = dataset / "replays"
    assert _run(dataset, output, "--check-only") == 2
    assert not output.exists()


def test_replay_refuses_to_overwrite_previous_report(tmp_path):
    dataset = _write_dataset(tmp_path)
    output = tmp_path / "replays"
    assert _run(dataset, output, "--check-only") == 0
    original = (output / "replay_report.json").read_bytes()
    assert _run(dataset, output, "--check-only") == 2
    assert (output / "replay_report.json").read_bytes() == original


def test_resolves_lerobot_paths_and_camera_keys(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path)
    info, episodes = replay.load_dataset_metadata(dataset_path)

    assert [episode["episode_index"] for episode in episodes] == [0, 1]
    assert replay.get_video_keys(info) == list(CAMERA_KEYS)
    assert replay.resolve_camera_keys(
        ["left_wrist", CAMERA_KEYS[0], "left_wrist"],
        CAMERA_KEYS,
    ) == [CAMERA_KEYS[1], CAMERA_KEYS[0]]
    assert replay.get_parquet_path(dataset_path, info, 1001) == (
        dataset_path / "data/chunk-001/episode_001001.parquet"
    )
    assert replay.get_video_path(dataset_path, info, 1001, CAMERA_KEYS[0]) == (
        dataset_path / "videos" / "chunk-001" / CAMERA_KEYS[0] / "episode_001001.mp4"
    )


def test_check_only_writes_passing_report_without_replay(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "check_only"

    status = _run(dataset_path, output_dir, "--check-only")

    assert status == 0
    report = _read_report(output_dir)
    assert report["summary"] == {
        "selected_episodes": 1,
        "reliable_episodes": 1,
        "failed_episodes": 0,
        "dataset_errors": 0,
        "passed": True,
    }
    assert report["selection"]["episode_indices"] == [0]
    assert report["selection"]["camera_keys"] == list(CAMERA_KEYS)
    assert not list(output_dir.glob("*_replay.mp4"))


def test_replay_tiles_both_cameras_with_expected_timing_and_size(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "rendered"

    status = _run(
        dataset_path,
        output_dir,
        "--episodes",
        "0",
        "--no-overlay",
    )

    assert status == 0
    report = _read_report(output_dir)
    replay_path = Path(report["episodes"][0]["replay_video"])
    assert replay_path.is_file()

    capture = cv2.VideoCapture(str(replay_path))
    assert capture.isOpened()
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAME_COUNT
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS, abs=0.01)
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * FRAME_WIDTH
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == FRAME_HEIGHT
        ok, first_frame = capture.read()
        assert ok
    finally:
        capture.release()

    left_tile = first_frame[:, :FRAME_WIDTH]
    right_tile = first_frame[:, FRAME_WIDTH:]
    assert left_tile[..., 2].mean() > left_tile[..., 1].mean()
    assert right_tile[..., 1].mean() > right_tile[..., 2].mean()


@pytest.mark.parametrize("corruption", ["nan", "video_length"])
def test_corrupt_episode_fails_report_and_returns_one(
    tmp_path: Path,
    corruption: str,
):
    dataset_path = _write_dataset(tmp_path)
    if corruption == "nan":
        parquet_path = _parquet_path(dataset_path, 0)
        frame = pd.read_parquet(parquet_path)
        state = frame["observation.state"].tolist()
        state[1] = [np.nan, 0.5]
        frame["observation.state"] = state
        frame.to_parquet(parquet_path, index=False)
        expected_error = "non-finite values"
    else:
        _write_video(
            _video_path(dataset_path, 0, CAMERA_KEYS[1]),
            frame_count=FRAME_COUNT - 1,
            color=(10, 220, 20),
        )
        expected_error = f"video has {FRAME_COUNT - 1} frames; expected {FRAME_COUNT}"

    output_dir = tmp_path / f"failed_{corruption}"
    status = _run(
        dataset_path,
        output_dir,
        "--episodes",
        "0",
        "--check-only",
    )

    assert status == 1
    report = _read_report(output_dir)
    assert report["summary"]["passed"] is False
    assert report["summary"]["failed_episodes"] == 1
    assert any(expected_error in error for error in report["episodes"][0]["errors"])


def test_discarded_episodes_are_excluded_by_default_and_can_be_included(
    tmp_path: Path,
):
    dataset_path = _write_dataset(tmp_path)
    default_output = tmp_path / "discard_default"

    assert _run(dataset_path, default_output, "--all", "--check-only") == 0
    default_report = _read_report(default_output)
    assert default_report["selection"]["episode_indices"] == [0]

    included_output = tmp_path / "discard_included"
    assert (
        _run(
            dataset_path,
            included_output,
            "--all",
            "--include-discarded",
            "--check-only",
        )
        == 1
    )
    included_report = _read_report(included_output)
    assert included_report["selection"]["episode_indices"] == [0, 1]
    discarded_check = included_report["episodes"][1]
    assert discarded_check["discarded"] is True
    assert any(
        "episode is marked discarded" in error for error in discarded_check["errors"]
    )
