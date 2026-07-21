import json
from types import SimpleNamespace

from robocasa.scripts import collect_sonic_demos as collector


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
