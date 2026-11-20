import numpy as np
import pytest
import robosuite

from robocasa.utils.config_utils import refactor_composite_controller_config
from robocasa.utils.eval_utils import create_eval_env


@pytest.mark.parametrize("controller", ["OSC_POSE", "JOINT_POSITION"])
def test_eval_controller_runs_with_existing_composite_conversion(
    monkeypatch, controller
):
    captured = {}
    monkeypatch.setattr(robosuite, "make", lambda **kwargs: captured.update(kwargs))

    create_eval_env("PnPCounterToCab", robots="Panda", controllers=controller)

    assert captured["robots"] == "Panda"
    assert captured["controller_configs"]["type"] == controller
    config = refactor_composite_controller_config(
        captured["controller_configs"], "Panda", ["right"]
    )
    assert config["body_parts"]["right"]["type"] == controller
    monkeypatch.undo()
    env = robosuite.make(
        "Lift",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    try:
        obs = env.reset()
        assert np.isfinite(obs["robot0_joint_pos"]).all()
        obs, reward, _, _ = env.step(np.zeros(env.action_dim))
        assert np.isfinite(obs["robot0_joint_pos"]).all()
        assert np.isfinite(reward)
    finally:
        env.close()


def test_eval_defaults_preserve_mobile_robot_and_controller(monkeypatch):
    captured = {}
    monkeypatch.setattr(robosuite, "make", lambda **kwargs: captured.update(kwargs))

    create_eval_env("PnPCounterToCab")

    assert captured["robots"] == "PandaMobile"
    config = refactor_composite_controller_config(
        captured["controller_configs"], "PandaOmron", ["right"]
    )
    assert config["body_parts"]["right"]["type"] == "OSC_POSE"
    assert "base" in config["body_parts"]
