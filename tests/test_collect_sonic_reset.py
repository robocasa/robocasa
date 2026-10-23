import json
from types import SimpleNamespace

import numpy as np

from robocasa.scripts import collect_sonic_demos as collector


def test_episode_event_priority_matches_exporter_abort_first():
    assert collector._resolve_episode_event(
        {"c"}, {"toggle_data_abort"}, recording=False
    ) == (None, True)
    assert collector._resolve_episode_event(
        {"k"}, {"toggle_data_abort"}, recording=True
    ) == ("discard", True)
    assert collector._resolve_episode_event(
        {"x"}, {"toggle_data_collection"}, recording=True
    ) == ("discard", False)
    assert collector._resolve_episode_event(
        set(), {"toggle_data_collection"}, recording=True
    ) == ("save", True)


def test_instruction_snapshot_is_single_read_and_frozen(capsys):
    calls = []

    class FakeBase:
        def get_ep_meta(self):
            calls.append("get")
            return {"lang": "Pick up the cup and bowl."}

        def set_ep_meta(self, metadata):
            calls.append(("set", metadata.copy()))

    metadata, instruction = collector._snapshot_and_print_instruction(FakeBase())

    assert metadata == {"lang": "Pick up the cup and bowl."}
    assert instruction == "Pick up the cup and bowl."
    assert calls == ["get", ("set", metadata)]
    assert "Instruction: Pick up the cup and bowl." in capsys.readouterr().out


def test_collection_wrapper_reuses_forwarded_episode_metadata(monkeypatch, tmp_path):
    expected_metadata = {"lang": "sampled instruction", "layout_id": 1}
    set_calls = []

    class FakeState:
        def flatten(self):
            return [1.0, 2.0]

    class FakeEnv:
        model = SimpleNamespace(get_xml=lambda: "<mujoco />")
        sim = SimpleNamespace(get_state=lambda: FakeState())

        def get_ep_meta(self):
            raise AssertionError("episode metadata must not be sampled twice")

        def set_ep_meta(self, metadata):
            set_calls.append(metadata)

    wrapper = object.__new__(collector.SonicDataCollectionWrapper)
    wrapper.env = FakeEnv()
    wrapper.directory = str(tmp_path)
    wrapper.has_interaction = False
    wrapper.states = []
    wrapper.integration_states = []
    wrapper.action_infos = []
    monkeypatch.setattr(wrapper, "_integration_state_for", lambda env: np.array([3.0]))

    wrapper.start_episode_from_current_state(ep_meta=expected_metadata)
    wrapper._on_first_interaction()

    assert set_calls == [expected_metadata]
    assert wrapper._current_task_instance_xml == "<mujoco />"
    with open(f"{wrapper.ep_directory}/ep_meta.json", encoding="utf-8") as stream:
        assert json.load(stream) == expected_metadata


def test_make_env_configures_and_records_pgs_solver(monkeypatch):
    captured = {}
    fake_env = object()

    def fake_make(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return fake_env

    monkeypatch.setattr(collector.robosuite, "make", fake_make)
    args = SimpleNamespace(
        environment="CloseDishwasher",
        robot="SonicG1",
        render_camera="robot0_head_camera",
        layout=1,
        style=1,
        control_freq=200,
        sim_dt=0.005,
        post_action_hz=20.0,
        render_hz=20.0,
        floor_friction=1.0,
        floor_torsion=0.005,
    )

    env, env_kwargs = collector.make_env(args, cfg={"type": "SONIC_WBC"})

    assert env is fake_env
    assert captured["mujoco_solver"] == "PGS"
    assert env_kwargs["mujoco_solver"] == "PGS"
    assert json.loads(json.dumps(env_kwargs))["mujoco_solver"] == "PGS"
    assert json.loads(collector._sonic_runtime_json(args))["mujoco_solver"] == "PGS"


def test_reset_with_retry_clears_episode_metadata_before_reset(monkeypatch):
    events = []
    observation = {"state": "reset"}

    class FakeEnvironment:
        def reset(self):
            events.append("reset")
            return observation

    class FakeBase:
        def unset_ep_meta(self):
            events.append("unset_ep_meta")

    monkeypatch.setattr(
        collector,
        "_apply_runtime",
        lambda base, args: events.append("apply_runtime"),
    )

    result = collector.reset_with_retry(
        FakeEnvironment(),
        FakeBase(),
        SimpleNamespace(),
    )

    assert result is observation
    assert events == ["unset_ep_meta", "reset", "apply_runtime"]


def test_reset_with_retry_clears_episode_metadata_before_every_attempt(monkeypatch):
    events = []
    observation = {"state": "reset"}
    reset_attempts = 0

    class FakeEnvironment:
        def reset(self):
            nonlocal reset_attempts
            reset_attempts += 1
            events.append("reset")
            if reset_attempts == 1:
                raise collector.mujoco.FatalError("rank-deficient Hessian")
            return observation

    class FakeBase:
        def unset_ep_meta(self):
            events.append("unset_ep_meta")

    monkeypatch.setattr(
        collector,
        "_apply_runtime",
        lambda base, args: events.append("apply_runtime"),
    )

    result = collector.reset_with_retry(
        FakeEnvironment(),
        FakeBase(),
        SimpleNamespace(),
        tries=2,
    )

    assert result is observation
    assert events == [
        "unset_ep_meta",
        "reset",
        "unset_ep_meta",
        "reset",
        "apply_runtime",
    ]
