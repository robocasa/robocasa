import xml.etree.ElementTree as ET

import numpy as np

import robocasa  # noqa: F401  Registers RoboCasa environments with robosuite.
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_sonic_g1_demos import SONIC_CFG

from robocasa.utils.env_utils import compute_robot_base_placement_pose


LEFT_OFFSET_METERS = 0.50


def _make_env():
    return robosuite.make(
        "SlideDishwasherRack",
        robots=["SonicG1"],
        controller_configs=load_composite_controller_config(
            controller=SONIC_CFG,
            robot="SonicG1",
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        layout_ids=1,
        style_ids=1,
        control_freq=200,
        initialization_noise=None,
        seed=7,
    )


def _xml_pelvis_position(env):
    root = ET.fromstring(env.model.get_xml())
    pelvis = next(
        body for body in root.iter("body") if body.get("name") == "robot0_pelvis"
    )
    return np.fromstring(pelvis.attrib["pos"], sep=" ")


def test_slide_dishwasher_rack_applies_default_left_offset_to_sonic_root():
    env = None
    try:
        env = _make_env()
        env.reset()

        unshifted_anchor, base_ori = compute_robot_base_placement_pose(
            env,
            ref_fixture=env.dishwasher,
        )
        yaw = base_ori[2]
        expected_delta = LEFT_OFFSET_METERS * np.array([-np.sin(yaw), np.cos(yaw), 0.0])

        np.testing.assert_allclose(
            env.init_robot_base_pos_anchor - unshifted_anchor,
            expected_delta,
            atol=1e-7,
        )
        assert env.robot_init_offset == (-LEFT_OFFSET_METERS, 0.0)
        np.testing.assert_allclose(
            env.get_ep_meta()["init_robot_base_pos"],
            env.init_robot_base_pos,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _xml_pelvis_position(env),
            env.init_robot_base_pos_anchor,
            atol=1e-7,
        )

        env.step(np.zeros(env.action_spec[0].shape))
        assert np.isfinite(env.sim.data.qpos).all()
        assert np.isfinite(env.sim.data.ctrl).all()
    finally:
        if env is not None:
            env.close()


def test_slide_dishwasher_rack_replays_with_the_fixed_default_left_offset():
    source = restored = None
    try:
        source = _make_env()
        source.reset()
        metadata = source.get_ep_meta()

        assert metadata["robot_init_offset"] == [-LEFT_OFFSET_METERS, 0.0]

        restored = _make_env()
        restored.set_ep_meta(metadata)
        restored.reset()

        np.testing.assert_allclose(
            restored.init_robot_base_pos_anchor,
            source.init_robot_base_pos_anchor,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _xml_pelvis_position(restored),
            restored.init_robot_base_pos_anchor,
            atol=1e-7,
        )
    finally:
        if source is not None:
            source.close()
        if restored is not None:
            restored.close()
