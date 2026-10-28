import json
from pathlib import Path

import h5py
import numpy as np
import pytest

# The CLI's data checks do not require importing the simulation package.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "evaluate_sonic_raw_under_test",
    Path(__file__).resolve().parents[1] / "robocasa/scripts/evaluate_sonic_raw.py",
)
_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _script
_spec.loader.exec_module(_script)

from evaluate_sonic_raw_under_test import (
    EpisodeCandidate,
    EpisodeData,
    EvaluationError,
    discover_episodes,
    evaluate_candidate,
    evaluate_candidate_isolated,
    main,
    load_episode,
    normalize_success,
    quarantine_failed_episodes,
    replay_success_checks,
)


def _write_episode(
    episode_dir: Path,
    *,
    task: str = "LoadDishwasher",
    hdf5_states: np.ndarray | None = None,
    raw_states: np.ndarray | None = None,
) -> Path:
    episode_dir.mkdir(parents=True)
    if hdf5_states is None:
        hdf5_states = np.array([[0.0, 1.0], [0.1, 2.0]])
    ep_meta = {"layout_id": 5, "style_id": 1, "lang": "test instruction"}
    model_xml = "<mujoco/>"
    hdf5_path = episode_dir / "ep_demo.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["env"] = task
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": task,
                "env_kwargs": {"control_freq": 10, "robots": ["SonicG1"]},
            }
        )
        data.attrs["sonic_runtime"] = json.dumps(
            {"control_freq": 10, "post_action_freq": 2, "sim_dt": 0.005}
        )
        demo = data.create_group("demo_1")
        demo.attrs["ep_meta"] = json.dumps(ep_meta)
        demo.attrs["model_file"] = model_xml
        demo.create_dataset("states", data=hdf5_states)
    (episode_dir / "ep_meta.json").write_text(json.dumps(ep_meta))
    (episode_dir / "model.xml").write_text(model_xml)
    if raw_states is not None:
        np.savez(episode_dir / "state_10_5.npz", states=raw_states)
    return hdf5_path


def _episode_data(states: np.ndarray, *, post_action_freq: int = 2) -> EpisodeData:
    candidate = EpisodeCandidate(
        episode_dir=Path("/dataset/Task_sonic/ep_1"),
        hdf5_path=Path("/dataset/Task_sonic/ep_1/ep_demo.hdf5"),
        task_hint="Task",
    )
    return EpisodeData(
        candidate=candidate,
        task="Task",
        env_meta={"env_name": "Task", "env_kwargs": {}},
        ep_meta={},
        model_xml="<mujoco/>",
        states=states,
        state_source="test",
        state_files=[],
        hdf5_state_count=len(states),
        control_freq=10,
        post_action_freq=post_action_freq,
        sonic_runtime_text=None,
        sonic_gains_text=None,
    )


class _FakeSim:
    def __init__(self) -> None:
        self.state = None

    def set_state_from_flattened(self, state):
        self.state = np.asarray(state)

    def forward(self):
        pass


class _FakeEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.events = []
        self.timestep = 0
        self.cur_time = 0.0

    def _check_success(self):
        value = bool(self.sim.state[1])
        self.events.append(("check", self.timestep, value))
        return {"task": value}

    def update_state(self):
        self.events.append(("update", self.timestep))


class _DiscardOnlyEnvCache:
    def __init__(self) -> None:
        self.discarded = False

    def discard(self) -> None:
        self.discarded = True


class _RuntimeEvaluationErrorCache(_DiscardOnlyEnvCache):
    def get(self, episode):
        raise EvaluationError("environment cannot restore recorded model")


def test_discovery_supports_reorganized_direct_episode_layout(tmp_path):
    ep_2 = tmp_path / "LoadDishwasher_sonic" / "ep_2"
    ep_1 = tmp_path / "LoadDishwasher_sonic" / "ep_1"
    _write_episode(ep_2)
    _write_episode(ep_1)

    candidates = discover_episodes(tmp_path)

    assert [candidate.episode_dir.name for candidate in candidates] == [
        "ep_1",
        "ep_2",
    ]
    assert {candidate.task_hint for candidate in candidates} == {"LoadDishwasher"}


def test_discovery_reports_missing_hdf5_without_following_symlink(tmp_path):
    task_dir = tmp_path / "LoadDishwasher_sonic"
    missing = task_dir / "ep_missing"
    missing.mkdir(parents=True)
    target = tmp_path / "outside" / "ep_target"
    target.mkdir(parents=True)
    (task_dir / "ep_link").symlink_to(target, target_is_directory=True)

    candidates = discover_episodes(tmp_path)
    by_name = {candidate.episode_dir.name: candidate for candidate in candidates}

    assert by_name["ep_missing"].hdf5_path is None
    assert by_name["ep_link"].discovery_error == "episode directory is a symlink"


def test_load_episode_prefers_raw_true_terminal_state(tmp_path):
    episode_dir = tmp_path / "LoadDishwasher_sonic" / "ep_1"
    hdf5_states = np.array([[0.0, 0.0], [0.1, 0.0]])
    raw_states = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 1.0]])
    hdf5_path = _write_episode(
        episode_dir,
        hdf5_states=hdf5_states,
        raw_states=raw_states,
    )

    episode = load_episode(
        EpisodeCandidate(episode_dir, hdf5_path, "LoadDishwasher")
    )

    assert episode.state_source == "raw_npz/true_terminal_state"
    np.testing.assert_array_equal(episode.states, raw_states)
    assert episode.hdf5_state_count == 2


def test_load_episode_rejects_misaligned_raw_state_count(tmp_path):
    episode_dir = tmp_path / "LoadDishwasher_sonic" / "ep_1"
    hdf5_path = _write_episode(
        episode_dir,
        raw_states=np.array([[0.0, 0.0], [0.1, 1.0]]),
    )

    with pytest.raises(EvaluationError, match="raw state count"):
        load_episode(EpisodeCandidate(episode_dir, hdf5_path, "LoadDishwasher"))


def test_structurally_invalid_episode_is_distinct_from_runtime_error(tmp_path):
    episode_dir = tmp_path / "LoadDishwasher_sonic" / "ep_invalid"
    hdf5_path = _write_episode(episode_dir)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        del hdf5_file["data/demo_1/states"]
    cache = _DiscardOnlyEnvCache()

    result = evaluate_candidate(
        EpisodeCandidate(episode_dir, hdf5_path, "LoadDishwasher"),
        cache,
        success_mode="any",
    )

    assert result["status"] == "invalid"
    assert result["error"].startswith("EvaluationError:")
    assert cache.discarded is True


def test_runtime_evaluation_error_is_not_classified_as_invalid(tmp_path):
    episode_dir = tmp_path / "LoadDishwasher_sonic" / "ep_runtime_error"
    hdf5_path = _write_episode(episode_dir)
    cache = _RuntimeEvaluationErrorCache()

    result = evaluate_candidate(
        EpisodeCandidate(episode_dir, hdf5_path, "LoadDishwasher"),
        cache,
        success_mode="any",
    )

    assert result["status"] == "error"
    assert result["error"].startswith("EvaluationError:")
    assert cache.discarded is True


def test_isolated_signal_is_aggregated_as_error(tmp_path, monkeypatch):
    episode_dir = tmp_path / "Task_sonic" / "ep_signal"
    episode_dir.mkdir(parents=True)
    hdf5_path = episode_dir / "ep_demo.hdf5"
    hdf5_path.write_bytes(b"placeholder")

    class FakeProcess:
        pid = 12345
        returncode = -11

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    result = evaluate_candidate_isolated(
        EpisodeCandidate(episode_dir, hdf5_path, "Task"),
        success_mode="any",
        seed=0,
        timeout_seconds=10,
        report_path=tmp_path / "child.json",
        log_path=tmp_path / "child.log",
    )

    assert result["status"] == "error"
    assert result["isolation"]["returncode"] == -11
    assert result["isolation"]["signal"] == 11
    assert "log_sha256" in result["isolation"]


def test_isolated_mode_aggregates_one_child_report(tmp_path):
    episode_dir = tmp_path / "LoadDishwasher_sonic" / "ep_invalid"
    hdf5_path = _write_episode(episode_dir)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        del hdf5_file["data/demo_1/states"]
    report_path = tmp_path / "evaluation.json"

    returncode = main(
        [
            "--dataset",
            str(episode_dir),
            "--task",
            "LoadDishwasher",
            "--success-mode",
            "any",
            "--seed",
            "7",
            "--isolate-episodes",
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    assert returncode == 0
    assert report["status"] == "complete"
    assert report["summary"] == {
        "total": 1,
        "passed": 0,
        "failed": 0,
        "invalid": 1,
        "error": 0,
        "missing_hdf5": 0,
    }
    assert report["results"][0]["status"] == "invalid"
    assert report["execution"]["episode_process_isolation"] is True
    assert report["execution"]["episode_timeout_seconds"] == 300.0
    child_dir = Path(report["execution"]["child_reports_dir"])
    assert len(list(child_dir.glob("*.json"))) == 1
    assert len(list(child_dir.glob("*.log"))) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (np.bool_(False), False),
        ({"task": 1}, True),
        ({"success": 0}, False),
    ],
)
def test_normalize_success(value, expected):
    assert normalize_success(value) is expected


def test_replay_uses_recorded_cadence_check_then_update_and_terminal_probe():
    # At 10 Hz with post_action_freq=2, t=0.2 is a normal checkpoint. The
    # terminal t=0.3 is additionally probed and is successful.
    episode = _episode_data(
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
            ]
        )
    )
    env = _FakeEnv()

    result = replay_success_checks(env, episode, success_mode="any")

    assert result["success"] is True
    assert result["checkpoint_count"] == 2
    assert result["terminal_probe_added"] is True
    assert env.events == [
        ("check", 2, False),
        ("update", 2),
        ("check", 3, True),
        ("update", 3),
    ]


def test_success_mode_final_differs_from_any():
    episode = _episode_data(
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 1.0],
                [0.3, 0.0],
                [0.4, 0.0],
            ]
        )
    )

    assert replay_success_checks(_FakeEnv(), episode, success_mode="any")[
        "success"
    ]
    assert not replay_success_checks(_FakeEnv(), episode, success_mode="final")[
        "success"
    ]


def test_cleanup_moves_only_failed_episode_to_recoverable_quarantine(tmp_path):
    dataset_root = tmp_path / "sonic_raw"
    failed = dataset_root / "Task_sonic" / "ep_failed"
    passed = dataset_root / "Task_sonic" / "ep_passed"
    failed.mkdir(parents=True)
    passed.mkdir(parents=True)
    report_path = tmp_path / "report.json"
    quarantine = tmp_path / "sonic_raw.failed-test"
    report = {
        "summary": {"total": 2, "passed": 1, "failed": 1, "error": 0, "missing_hdf5": 0},
        "cleanup": {},
        "results": [
            {"status": "failed", "task": "Task", "episode_dir": str(failed)},
            {"status": "passed", "task": "Task", "episode_dir": str(passed)},
        ],
    }

    quarantine_failed_episodes(
        report,
        dataset_root=dataset_root,
        quarantine_root=quarantine,
        report_path=report_path,
    )

    assert not failed.exists()
    assert passed.is_dir()
    assert (quarantine / "Task_sonic" / "ep_failed").is_dir()
    saved = json.loads(report_path.read_text())
    assert saved["cleanup"]["status"] == "complete"


def test_cleanup_also_moves_deterministically_invalid_episode(tmp_path):
    dataset_root = tmp_path / "sonic_raw"
    invalid = dataset_root / "Task_sonic" / "ep_invalid"
    invalid.mkdir(parents=True)
    report_path = tmp_path / "report.json"
    quarantine = tmp_path / "sonic_raw.failed-test"
    report = {
        "summary": {
            "total": 1,
            "passed": 0,
            "failed": 0,
            "invalid": 1,
            "error": 0,
            "missing_hdf5": 0,
        },
        "cleanup": {},
        "results": [
            {
                "status": "invalid",
                "task": "Task",
                "episode_dir": str(invalid),
                "error": "EvaluationError: demo has no states dataset",
            }
        ],
    }

    quarantine_failed_episodes(
        report,
        dataset_root=dataset_root,
        quarantine_root=quarantine,
        report_path=report_path,
    )

    assert not invalid.exists()
    assert (quarantine / "Task_sonic" / "ep_invalid").is_dir()
    assert report["cleanup"]["status"] == "complete"


def test_cleanup_is_fail_closed_when_any_evaluation_errored(tmp_path):
    dataset_root = tmp_path / "sonic_raw"
    failed = dataset_root / "Task_sonic" / "ep_failed"
    failed.mkdir(parents=True)
    report_path = tmp_path / "report.json"
    report = {
        "summary": {"total": 2, "passed": 0, "failed": 1, "error": 1, "missing_hdf5": 0},
        "cleanup": {},
        "results": [
            {"status": "failed", "task": "Task", "episode_dir": str(failed)},
            {"status": "error", "task": "Task", "episode_dir": str(dataset_root / "Task_sonic" / "ep_error")},
        ],
    }

    quarantine_failed_episodes(
        report,
        dataset_root=dataset_root,
        quarantine_root=tmp_path / "sonic_raw.failed-test",
        report_path=report_path,
    )

    assert failed.is_dir()
    assert report["cleanup"]["status"] == "skipped_due_to_evaluation_errors"


def test_evaluator_refuses_report_inside_dataset(tmp_path):
    root = tmp_path / "raw"
    _write_episode(root / "LoadDishwasher_sonic" / "ep_1")
    report = root / "report.json"
    assert main(["--dataset", str(root), "--report", str(report)]) == 2
    assert not report.exists()
