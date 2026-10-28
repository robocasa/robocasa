import json
import os
from pathlib import Path

import pytest

# The CLI's data checks do not require importing the simulation package.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "align_sonic_success_lerobot_under_test",
    Path(__file__).resolve().parents[1] / "robocasa/scripts/align_sonic_success_lerobot.py",
)
_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _script
_spec.loader.exec_module(_script)

from align_sonic_success_lerobot_under_test import (
    AlignmentError,
    align_by_completion_time,
    build_selection_plan,
    cleaner_python_path,
    discover_lerobot_episodes,
    discover_raw_episodes,
    load_replay_statuses,
)


def _write_raw_episode(
    raw_root: Path,
    *,
    task: str,
    episode: str,
    end_timestamp: str,
    with_hdf5: bool,
) -> Path:
    task_dir = raw_root / f"2026-08-19-12-00-00_{task}_sonic"
    episode_dir = task_dir / "episodes" / episode
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / f"state_{end_timestamp}.npz").write_bytes(b"state")
    if with_hdf5:
        (episode_dir / "ep_demo.hdf5").write_bytes(b"hdf5")
    return episode_dir


def _write_lerobot_dataset(
    root: Path,
    *,
    task: str,
    episode_mtimes: list[float],
    discarded: tuple[int, ...] = (),
) -> Path:
    dataset = root / f"robocasa_{task}_g1_3cam"
    (dataset / "meta").mkdir(parents=True)
    info = {
        "total_episodes": len(episode_mtimes),
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "discarded_episode_indices": list(discarded),
        "features": {
            "action.motion_token": {"dtype": "float64", "shape": [64]},
            "observation.images.ego_view": {"dtype": "video"},
        },
    }
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    with (dataset / "meta" / "episodes.jsonl").open("w") as file:
        for index in range(len(episode_mtimes)):
            file.write(
                json.dumps({"episode_index": index, "length": 2, "tasks": []})
                + "\n"
            )
    for index, mtime in enumerate(episode_mtimes):
        parquet = (
            dataset
            / "data"
            / "chunk-000"
            / f"episode_{index:06d}.parquet"
        )
        video = (
            dataset
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{index:06d}.mp4"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"parquet")
        video.write_bytes(b"video")
        os.utime(parquet, (mtime, mtime))
    return dataset


def _write_replay_report(
    path: Path,
    *,
    task: str,
    statuses: dict[str, str],
    completed_at: str = "2026-09-01T12:00:00+08:00",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "success_mode": "any",
                "completed_at": completed_at,
                "results": [
                    {"task": task, "episode": episode, "status": status}
                    for episode, status in statuses.items()
                ],
            }
        )
    )
    return path


def test_latest_replay_result_supersedes_older_status_and_keeps_provenance(
    tmp_path,
):
    task = "TaskReplay"
    older = _write_replay_report(
        tmp_path / "older.json",
        task=task,
        statuses={"ep_1": "error"},
        completed_at="2026-09-01T11:00:00+08:00",
    )
    newer = _write_replay_report(
        tmp_path / "newer.json",
        task=task,
        statuses={"ep_1": "passed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )

    statuses, provenance = load_replay_statuses([newer, older], task)

    assert statuses == {"ep_1": "passed"}
    assert [item["path"] for item in provenance] == [
        str(newer.resolve()),
        str(older.resolve()),
    ]
    assert all(item["sha256"] for item in provenance)
    assert [item["completed_at"] for item in provenance] == [
        "2026-09-01T13:00:00+08:00",
        "2026-09-01T11:00:00+08:00",
    ]


def test_replay_result_completion_time_precedence(tmp_path):
    task = "TaskReplay"
    report = tmp_path / "precedence.json"
    report.write_text(
        json.dumps(
            {
                "completed_at": "2026-09-01T15:00:00+08:00",
                "results": [
                    {
                        "task": task,
                        "episode": "ep_1",
                        "status": "failed",
                        "completed_at": "2026-09-01T14:00:00+08:00",
                        "isolation": {
                            "completed_at": "2026-09-01T11:00:00+08:00"
                        },
                    },
                    {
                        "task": task,
                        "episode": "ep_1",
                        "status": "passed",
                        "completed_at": "2026-09-01T12:00:00+08:00",
                    },
                ],
            }
        )
    )

    statuses, _provenance = load_replay_statuses([report], task)

    assert statuses == {"ep_1": "passed"}


def test_conflicting_statuses_only_abort_at_same_latest_time(tmp_path):
    task = "TaskReplay"
    first = _write_replay_report(
        tmp_path / "first.json",
        task=task,
        statuses={"ep_1": "failed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )
    second = _write_replay_report(
        tmp_path / "second.json",
        task=task,
        statuses={"ep_1": "passed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )

    with pytest.raises(AlignmentError, match="conflicting latest replay statuses"):
        load_replay_statuses([first, second], task)


def test_plan_keeps_only_uniquely_matched_replay_passed_episode(tmp_path):
    task = "TaskA"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=False,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="200_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_3",
        end_timestamp="300_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(
        tmp_path, task=task, episode_mtimes=[103.0, 204.0, 305.0, 500.0]
    )
    report = _write_replay_report(
        tmp_path / "report.json",
        task=task,
        statuses={"ep_2": "passed", "ep_3": "failed"},
    )

    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    statuses, provenance = load_replay_statuses([report], task)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )
    plan = build_selection_plan(
        task=task,
        dataset=dataset,
        raw_episodes=raw,
        lerobot_episodes=lerobot,
        matches=matches,
        unmatched_lerobot=unmatched,
        replay_statuses=statuses,
        replay_provenance=provenance,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    assert plan["kept_source_episode_indices"] == [1]
    assert plan["excluded_source_episode_indices"] == [0, 2, 3]
    assert plan["extra_discard_episode_indices"] == [0, 2, 3]
    assert plan["unmatched_lerobot_episode_indices"] == [3]


def test_existing_discard_is_not_reintroduced(tmp_path):
    task = "TaskB"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="200_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(
        tmp_path, task=task, episode_mtimes=[102.0, 202.0], discarded=(0,)
    )
    report = _write_replay_report(
        tmp_path / "report.json",
        task=task,
        statuses={"ep_1": "passed", "ep_2": "passed"},
    )
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    statuses, provenance = load_replay_statuses([report], task)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    plan = build_selection_plan(
        task=task,
        dataset=dataset,
        raw_episodes=raw,
        lerobot_episodes=lerobot,
        matches=matches,
        unmatched_lerobot=unmatched,
        replay_statuses=statuses,
        replay_provenance=provenance,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    assert plan["kept_source_episode_indices"] == [1]
    assert plan["successful_but_source_discarded"] == [0]
    assert plan["extra_discard_episode_indices"] == []


def test_missing_replay_result_fails_closed(tmp_path):
    task = "TaskC"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(tmp_path, task=task, episode_mtimes=[102.0])
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    with pytest.raises(AlignmentError, match="missing replay results"):
        build_selection_plan(
            task=task,
            dataset=dataset,
            raw_episodes=raw,
            lerobot_episodes=lerobot,
            matches=matches,
            unmatched_lerobot=unmatched,
            replay_statuses={},
            replay_provenance=[],
            max_lag_seconds=15.0,
            ambiguity_margin_seconds=1.0,
        )


def test_ambiguous_timestamp_match_is_rejected(tmp_path):
    task = "TaskD"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="100_5",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(tmp_path, task=task, episode_mtimes=[102.0])
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)

    with pytest.raises(AlignmentError, match="ambiguous timestamp match"):
        align_by_completion_time(
            raw,
            lerobot,
            max_lag_seconds=15.0,
            ambiguity_margin_seconds=1.0,
        )


def test_cleaner_python_path_preserves_virtualenv_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.write_bytes(b"python")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    selected = cleaner_python_path(venv_python)

    assert selected == venv_python.absolute()
    assert selected != venv_python.resolve()


def _alignment_inputs(tmp_path):
    task = "TaskReplay"
    raw = tmp_path / "raw"
    _write_raw_episode(raw, task=task, episode="ep_1", end_timestamp="100_0", with_hdf5=True)
    dataset = _write_lerobot_dataset(tmp_path, task=task, episode_mtimes=[104.0])
    report = _write_replay_report(
        tmp_path / "replay.json", task=task, statuses={"ep_1": "passed"}
    )
    return raw, dataset, [
        "--raw-root", str(raw), "--lerobot-dataset", str(dataset),
        "--replay-report", str(report),
    ]


def test_alignment_plan_preserves_sources_and_refuses_report_overwrite(tmp_path):
    raw, dataset, args = _alignment_inputs(tmp_path)
    snapshots = {p: p.read_bytes() for root in (raw, dataset) for p in root.rglob("*") if p.is_file()}
    report = tmp_path / "alignment.json"
    assert _script.main([*args, "--report", str(report)]) == 0
    saved = report.read_bytes()
    with pytest.raises(AlignmentError, match="overwrite report"):
        _script.main([*args, "--report", str(report)])
    assert report.read_bytes() == saved
    assert {p: p.read_bytes() for root in (raw, dataset) for p in root.rglob("*") if p.is_file()} == snapshots


@pytest.mark.parametrize("inside", ["raw", "lerobot"])
def test_alignment_refuses_report_inside_source(tmp_path, inside):
    raw, dataset, args = _alignment_inputs(tmp_path)
    report = (raw if inside == "raw" else dataset) / "alignment.json"
    with pytest.raises(AlignmentError, match="outside the source"):
        _script.main([*args, "--report", str(report)])
    assert not report.exists()


def test_alignment_apply_requires_and_forwards_processor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    raw, dataset, args = _alignment_inputs(tmp_path)
    original_files = {
        path: path.read_bytes()
        for source in (raw, dataset)
        for path in source.rglob("*") if path.is_file()
    }
    monkeypatch.delenv("SONIC_DATASET_PROCESSOR", raising=False)
    with pytest.raises(AlignmentError, match="requires --processor-path"):
        _script.main([*args, "--apply"])
    processor = tmp_path / "sonic" / "process_dataset.py"
    commands = []

    def run_cleaner(command, check):
        commands.append(command)
        meta = dataset.with_name(dataset.name + "_replay_cleaned") / "meta"
        meta.mkdir(parents=True)
        (meta / "cleanup_report.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_script.subprocess, "run", run_cleaner)
    assert _script.main([
        *args, "--apply", "--processor-path", str(processor),
        "--report", str(tmp_path / "applied.json"),
    ]) == 0
    assert commands[0][0] == str(Path(sys.executable).absolute())
    index = commands[0].index("--processor-path")
    assert commands[0][index + 1] == str(processor)
    external_report = json.loads((tmp_path / "applied.json").read_text())
    assert external_report["matches"][0]["replay_status"] == "passed"
    assert external_report["matches"][0]["selection"] == "keep_replay_passed"
    assert external_report["cleaner_report_sha256"]
    output_meta = dataset.with_name(dataset.name + "_replay_cleaned") / "meta"
    assert {path.name for path in output_meta.iterdir()} == {"cleanup_report.json"}
    assert (output_meta / "cleanup_report.json").read_text() == "{}"
    assert {
        path: path.read_bytes()
        for source in (raw, dataset)
        for path in source.rglob("*") if path.is_file()
    } == original_files
