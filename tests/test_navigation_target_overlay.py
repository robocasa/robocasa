from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robocasa.utils.navigation_target_overlay import NavigationTargetOverlay
from robocasa.utils.sonic_vla_streaming import RoboCasaVLACameraPublisher, VLACameraConfig


XML = """
<mujoco>
  <worldbody>
    <light pos="0 -2 4"/>
    <geom type="plane" size="4 4 .1" group="1" rgba=".2 .2 .2 1"/>
    <body pos="0 0 .5"><freejoint/>
      <geom type="box" size=".2 .3 .5" group="1" rgba=".6 .6 .6 1"/>
    </body>
    <camera name="robot0_head_camera" pos="0 -3 1.5" xyaxes="1 0 0 0 1 3"/>
    <camera name="robot0_left_wrist_camera" pos="-.3 -3 1.5" xyaxes="1 0 0 0 1 3"/>
    <camera name="robot0_right_wrist_camera" pos=".3 -3 1.5" xyaxes="1 0 0 0 1 3"/>
  </worldbody>
</mujoco>
"""


class Fixture:
    def __init__(self, name="fridge", x=0, angle=0):
        self.nat_lang = name
        self.x = x
        self.angle = angle

    def get_bbox_points(self):
        points = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
            [-1, 1, 1], [1, 1, 1], [1, 1, -1], [1, -1, 1],
        ]) * [.2, .3, .5]
        c, s = np.cos(self.angle), np.sin(self.angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return points @ rotation.T + [self.x, 0, .5]


class Viewer:
    def __init__(self, scene):
        self.user_scn = scene
        self.texts = None
        self.sync_count = 0
        self.running = True

    def lock(self):
        return nullcontext()

    def sync(self):
        self.sync_count += 1

    def set_texts(self, texts):
        self.texts = texts

    def clear_texts(self):
        self.texts = None

    def is_running(self):
        return self.running


def _env():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)),
        viewer=SimpleNamespace(viewer=None), target_fixture=Fixture(),
    )


def test_target_updates_after_delayed_viewer_creation_and_resets():
    env = _env()
    overlay = NavigationTargetOverlay()
    assert not overlay.update(env)  # robosuite opens the window on its first step
    scene = mujoco.MjvScene(env.sim.model._model, maxgeom=30)
    scene.ngeom = 1
    scene.geoms[0].label = "existing annotation"
    env.viewer.viewer = viewer = Viewer(scene)
    assert overlay.update(env)
    assert scene.ngeom == 14 and viewer.texts[-1] == "fridge"
    assert scene.geoms[13].label == "TARGET: fridge"
    for _ in range(100):
        assert not overlay.update(env)
    assert scene.ngeom == 14 and viewer.sync_count == 1

    # Soft reset: same window, new rotated destination, no accumulated markers.
    env.target_fixture = Fixture("toaster", x=2, angle=np.pi / 2)
    assert overlay.update(env)
    assert scene.ngeom == 14 and viewer.texts[-1] == "toaster"
    assert scene.geoms[13].label == "TARGET: toaster"
    np.testing.assert_allclose(scene.geoms[13].pos[0], 2)
    assert scene.geoms[0].label == "existing annotation"

    # k/x hard reset: MuJoCo model and passive viewer are both recreated.
    viewer.running = False
    env = _env()
    env.target_fixture = Fixture("sink", x=-2)
    new_scene = mujoco.MjvScene(env.sim.model._model, maxgeom=30)
    env.viewer.viewer = new_viewer = Viewer(new_scene)
    assert overlay.update(env)
    assert new_scene.ngeom == 13 and new_viewer.texts[-1] == "sink"
    assert new_scene.geoms[12].label == "TARGET: sink"
    overlay.close()
    assert new_scene.ngeom == 0 and new_viewer.texts is None


def test_full_viewer_buffer_is_preserved():
    env = _env()
    scene = mujoco.MjvScene(env.sim.model._model, maxgeom=5)
    scene.ngeom = 1
    scene.geoms[0].label = "existing annotation"
    env.viewer.viewer = Viewer(scene)
    overlay = NavigationTargetOverlay()
    with pytest.warns(UserWarning, match="buffer is full"):
        assert not overlay.update(env)
    assert not overlay.update(env)
    assert scene.ngeom == 1 and scene.geoms[0].label == "existing annotation"
    assert env.viewer.viewer.texts is None


def test_visible_operator_marker_leaves_recorded_cameras_model_and_state_identical():
    env = _env()
    model, data = env.sim.model._model, env.sim.data._data
    publisher = RoboCasaVLACameraPublisher(VLACameraConfig(
        camera_names=("robot0_head_camera", "robot0_left_wrist_camera", "robot0_right_wrist_camera"),
        output_keys=("ego_view", "left_wrist", "right_wrist"), width=160, height=120,
    ))
    overlay = NavigationTargetOverlay()
    operator = mujoco.Renderer(model, height=120, width=160)
    try:
        clean_before = publisher._render_images(env)
        model_before = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=model_before)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_before = np.zeros(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state_before, spec)

        # Render the operator scene with the same custom geometries that the
        # passive viewer merges from user_scn, without opening a desktop window.
        operator.update_scene(data, camera="robot0_head_camera")
        operator_before = operator.render().copy()
        env.viewer.viewer = Viewer(operator.scene)
        assert overlay.update(env)
        operator_after = operator.render().copy()
        assert np.count_nonzero(np.any(operator_before != operator_after, axis=2)) > 100

        clean_after = publisher._render_images(env)
        for camera in clean_before:
            np.testing.assert_array_equal(clean_before[camera], clean_after[camera])
        model_after = np.zeros_like(model_before)
        mujoco.mj_saveModel(model, buffer=model_after)
        np.testing.assert_array_equal(model_before, model_after)
        state_after = np.zeros_like(state_before)
        mujoco.mj_getState(model, data, state_after, spec)
        np.testing.assert_array_equal(state_before, state_after)
    finally:
        overlay.close()
        publisher.close()
        operator.close()
