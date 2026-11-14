from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from robosuite.models.arenas import Arena
from robosuite.utils import camera_utils as CU
from robosuite.utils.binding_utils import MjSim

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.utils import camera_utils as CamUtils


@pytest.fixture
def kitchen(tmp_path):
    xml = """<mujoco><asset/><worldbody>
        <body name="mobilebase0_support"><camera name="robot0_agentview_right" pos="0.4 0.1 2" quat="1 0 0 0" fovy="60"/></body>
        <body name="robot0_right_hand" pos="0 0 1">
            <joint name="slide" type="slide" axis="1 0 0"/>
            <geom type="sphere" size="0.01" mass="1"/>
            <camera name="robot0_eye_in_hand" pos="0.02 0 0" quat="1 0 0 0"/>
        </body>
        </worldbody><actuator/></mujoco>"""
    path = tmp_path / "arena.xml"
    path.write_text(xml)
    env = Kitchen.__new__(Kitchen)
    env.mujoco_arena = Arena(str(path))
    env.robots = [SimpleNamespace(name="PandaOmron")]
    env.use_cotraining_cameras = False
    env.randomize_cameras = False
    env.rng = np.random.default_rng(42)
    env.generative_textures = None
    env._ep_meta = {}
    return env, xml


@pytest.mark.parametrize("randomize", [False, True])
def test_recorded_camera_metadata_preserves_native_projection(kitchen, randomize):
    env, xml = kitchen
    recorded = {
        "robot0_agentview_right": {
            "parent_body": "mobilebase0_support",
            "pos": [0.4, 0.1, 2.0],
            "quat": [1, 0, 0, 0],
            "camera_attribs": {"fovy": "60"},
        },
        "robot0_eye_in_hand": {
            "parent_body": "robot0_right_hand",
            "pos": [0.02, 0, 0],
            "quat": [1, 0, 0, 0],
            "camera_attribs": {"fovy": "45"},
        },
    }
    env._ep_meta = {"cam_configs": deepcopy(recorded)}
    env.randomize_cameras = randomize
    rng_state = deepcopy(env.rng.bit_generator.state)
    CamUtils.set_cameras(env)
    sim = MjSim.from_xml_string(env.edit_model_xml(xml))
    sim.forward()
    camera_id = sim.model.camera_name2id("robot0_agentview_right")
    np.testing.assert_allclose(sim.data.cam_xpos[camera_id], [0.4, 0.1, 2])
    assert sim.model.cam_fovy[camera_id] == 60
    transform = CU.get_camera_transform_matrix(sim, "robot0_agentview_right", 128, 128)
    pixels = CU.project_points_from_world_to_camera(
        np.array([[0.1, -0.2, 0.25]]), transform, 128, 128
    )
    np.testing.assert_array_equal(
        pixels, [[83, 45]]
    )  # Existing helper returns row, column.
    wrist_id = sim.model.camera_name2id("robot0_eye_in_hand")
    np.testing.assert_allclose(sim.model.cam_pos[wrist_id], [0.02, 0, 0])
    sim.data.qpos[0] = 0.3
    sim.forward()
    np.testing.assert_allclose(sim.data.cam_xpos[wrist_id], [0.32, 0, 1], atol=1e-7)
    assert env.rng.bit_generator.state == rng_state
    env._cam_configs["robot0_agentview_right"]["pos"][0] = 9
    assert (
        env._ep_meta["cam_configs"] == recorded
    )  # Keep nested episode metadata immutable.


@pytest.mark.parametrize("randomize", [False, True])
def test_missing_camera_metadata_keeps_existing_sampling(kitchen, randomize):
    env, _ = kitchen
    env.randomize_cameras = randomize
    expected = CamUtils.get_robot_cam_configs("PandaOmron")
    CamUtils.set_cameras(env)
    assert env._cam_configs.keys() == expected.keys()
    camera = "robot0_agentview_right"
    if randomize:
        assert not np.allclose(env._cam_configs[camera]["pos"], expected[camera]["pos"])
    else:
        assert env._cam_configs == expected
