import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pytest

# The CLI's data checks do not require importing the simulation package.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "clean_sonic_dataset_under_test",
    Path(__file__).resolve().parents[1] / "robocasa/scripts/clean_sonic_dataset.py",
)
_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _script
_spec.loader.exec_module(_script)

from clean_sonic_dataset_under_test import (
    DatasetJob,
    build_mode_aware_stale_mask,
    filter_video_frames_streaming,
    inspect_video,
    main,
    remove_extra_discarded_episodes,
    scan_dataset,
    validate_processed_episode_indices,
)


def _write_scan_fixture(
    tmp_path: Path,
    discarded_episode_indices: tuple[int, ...] = (),
) -> Path:
    dataset_path = tmp_path / "dataset"
    metadata_path = dataset_path / "meta"
    parquet_path = dataset_path / "data" / "chunk-000"
    metadata_path.mkdir(parents=True)
    parquet_path.mkdir(parents=True)

    info = {
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 50,
        "chunks_size": 1000,
        "discarded_episode_indices": list(discarded_episode_indices),
        "features": {
            "teleop.smpl_pose": {"dtype": "float32", "shape": [2]},
            "teleop.stream_mode": {"dtype": "int64", "shape": [1]},
        },
    }
    (metadata_path / "info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    (metadata_path / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "length": 2}),
                json.dumps({"episode_index": 1, "length": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "teleop.smpl_pose": [[1.0, 2.0], [2.0, 3.0]],
            "teleop.stream_mode": [1, 1],
        }
    ).to_parquet(parquet_path / "episode_000000.parquet")
    return dataset_path


def _processed_episode(index: int) -> dict[str, Any]:
    return {"episode_meta": {"episode_index": index}}


def test_mode_aware_mask_preserves_non_pose_zero_frames():
    smpl_pose = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    stream_mode = np.array([1, 1, 1, 2, 0], dtype=np.int32)

    mask = build_mode_aware_stale_mask(smpl_pose, stream_mode)

    np.testing.assert_array_equal(mask, [False, True, True, False, False])


def test_mode_aware_mask_does_not_cross_stream_mode_boundary():
    smpl_pose = np.array(
        [[1.0, 2.0], [1.0, 2.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    stream_mode = np.array([2, 1, 1], dtype=np.int32)

    mask = build_mode_aware_stale_mask(smpl_pose, stream_mode)

    np.testing.assert_array_equal(mask, [False, False, True])


def test_mode_aware_mask_does_not_remove_natural_pose_repeat():
    smpl_pose = np.array(
        [[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]],
        dtype=np.float32,
    )
    stream_mode = np.array([1, 1, 1], dtype=np.int32)

    mask = build_mode_aware_stale_mask(smpl_pose, stream_mode)

    assert not mask.any()


def test_mode_aware_mask_validates_lengths():
    with pytest.raises(ValueError, match="lengths differ"):
        build_mode_aware_stale_mask(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
        )


def test_scan_allows_explicit_incomplete_episode_without_mutating_source(tmp_path):
    dataset_path = _write_scan_fixture(tmp_path)
    info_before = (dataset_path / "meta" / "info.json").read_bytes()
    episodes_before = (dataset_path / "meta" / "episodes.jsonl").read_bytes()
    job = DatasetJob(
        source=dataset_path,
        output=tmp_path / "dataset_cleaned",
        partial=tmp_path / ".dataset_cleaned.partial",
    )

    scan = scan_dataset(job, pose_modes=(1, 4), extra_discarded=(1,))

    assert scan.source_episodes == 2
    assert scan.source_frames == 5
    assert scan.extra_discarded_episode_indices == [1]
    assert scan.extra_discarded_episodes == 1
    assert scan.extra_discarded_frames == 3
    assert scan.expected_output_episodes == 1
    assert scan.expected_output_frames == 2
    assert (dataset_path / "meta" / "info.json").read_bytes() == info_before
    assert (
        dataset_path / "meta" / "episodes.jsonl"
    ).read_bytes() == episodes_before


def test_scan_rejects_unknown_or_already_discarded_extra_episode(tmp_path):
    dataset_path = _write_scan_fixture(tmp_path, discarded_episode_indices=(1,))
    job = DatasetJob(
        source=dataset_path,
        output=tmp_path / "dataset_cleaned",
        partial=tmp_path / ".dataset_cleaned.partial",
    )

    with pytest.raises(ValueError, match="not present"):
        scan_dataset(job, pose_modes=(1, 4), extra_discarded=(9,))
    with pytest.raises(ValueError, match="already marked discarded"):
        scan_dataset(job, pose_modes=(1, 4), extra_discarded=(1,))


def test_processed_episode_validation_accepts_only_explicit_skip(tmp_path):
    dataset_path = _write_scan_fixture(tmp_path)
    processor_output = [_processed_episode(0), _processed_episode(1)]
    filtered = remove_extra_discarded_episodes(processor_output, (1,))

    validate_processed_episode_indices(dataset_path, filtered, (1,))
    validate_processed_episode_indices(
        dataset_path,
        [_processed_episode(0)],
        (1,),
    )

    with pytest.raises(RuntimeError, match=r"missing=\[0\]"):
        validate_processed_episode_indices(dataset_path, [], (1,))


def test_extra_discard_cli_is_restricted_to_one_dataset():
    with pytest.raises(ValueError, match="exactly one input dataset"):
        main(
            [
                "/tmp/dataset_one",
                "/tmp/dataset_two",
                "--extra-discard-episode",
                "5",
                "--dry-run",
            ]
        )


def test_processor_must_be_explicit_but_dry_run_needs_none(monkeypatch):
    monkeypatch.delenv("SONIC_DATASET_PROCESSOR", raising=False)
    assert _script.parse_args(["dataset", "--dry-run"]).processor_path is None
    with pytest.raises(SystemExit) as error:
        _script.parse_args(["dataset"])
    assert error.value.code == 2
    monkeypatch.setenv("SONIC_DATASET_PROCESSOR", "/opt/sonic/process_dataset.py")
    assert _script.parse_args(["dataset"]).processor_path == Path(
        "/opt/sonic/process_dataset.py"
    )


def test_cleaner_refuses_output_that_is_another_input(tmp_path):
    source = _write_scan_fixture(tmp_path)
    other_source = source.with_name(source.name + "_cleaned")
    other_source.mkdir()
    with pytest.raises(ValueError, match="replace an input"):
        _script.build_jobs([source, other_source], "_cleaned", True)


def test_cleaner_refuses_output_symlink(tmp_path):
    source = _write_scan_fixture(tmp_path)
    source.with_name(source.name + "_cleaned").symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        _script.build_jobs([source], "_cleaned", True)


def test_streaming_video_filter_keeps_requested_frame_count(tmp_path):
    video_path = tmp_path / "episode.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("h264", rate=50)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for value in range(8):
        image = np.full((16, 16, 3), value * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    filter_video_frames_streaming(
        video_path,
        np.array([0, 2, 5, 7], dtype=np.int64),
        fps=50,
    )

    frame_count, width, height, fps = inspect_video(video_path)
    assert (frame_count, width, height, fps) == (4, 16, 16, 50.0)
