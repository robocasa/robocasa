"""The drawer side chosen by a task must also reach the robot's MJCF pose."""

from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.kitchen.atomic import kitchen_drawer
from robocasa.environments.kitchen.composite.measuring_ingredients.choose_measuring_cup import (
    ChooseMeasuringCup,
)
from robocasa.environments.kitchen.composite.tidying_cabinets_and_drawers.drawer_cake_sort import (
    DrawerCakeSort,
)


class _RobotModel:
    """Record the position and orientation written by Kitchen's real writer."""

    def __init__(self):
        self.positions = []
        self.orientations = []

    def set_base_xpos(self, position):
        self.positions.append(np.array(position, copy=True))

    def set_base_ori(self, orientation):
        self.orientations.append(np.array(orientation, copy=True))


def _task(monkeypatch, task_class, allowed_side):
    task = object.__new__(task_class)
    model = _RobotModel()
    task.robots = [SimpleNamespace(robot_model=model)]
    task.drawer = SimpleNamespace(width=0.8)
    task.rng = np.random.default_rng(4)
    task._point_outside_scene = lambda position: False
    task.check_fxtr_contact = lambda position: (
        allowed_side is None
        or (position[0] < 0) != (allowed_side == "left")
    )
    parent_calls = []

    def parent_load(self, *args, **kwargs):
        parent_calls.append((args, kwargs))
        self.init_robot_base_pos_anchor = np.array([0.0, -0.3, 0.793])
        self.init_robot_base_ori_anchor = np.array([0.0, 0.0, 0.25])
        self._write_robot_base_pose_to_model()

    def side_pose(env, ref_fixture, offset):
        return np.array([*offset, 0.793]), np.array([0.0, 0.0, 1.2])

    monkeypatch.setattr(kitchen_drawer.Kitchen, "_load_model", parent_load)
    monkeypatch.setattr(
        kitchen_drawer.EnvUtils, "compute_robot_base_placement_pose", side_pose
    )
    return task, model, parent_calls


@pytest.mark.parametrize(
    "task_class", [kitchen_drawer.OpenDrawer, ChooseMeasuringCup, DrawerCakeSort]
)
@pytest.mark.parametrize("allowed_side", ["left", "right"])
def test_selected_drawer_side_reaches_robot_model(monkeypatch, task_class, allowed_side):
    task, model, parent_calls = _task(monkeypatch, task_class, allowed_side)

    task._load_model("sentinel", attempt_num=4)

    expected_x = -0.6 if allowed_side == "left" else 0.6
    expected_drawer_side = "right" if allowed_side == "left" else "left"
    np.testing.assert_allclose(model.positions[-1], [expected_x, -0.1, 0.793])
    np.testing.assert_allclose(model.orientations[-1], [0.0, 0.0, 1.2])
    np.testing.assert_allclose(model.positions[-1], task.init_robot_base_pos_anchor)
    assert task.drawer_side == expected_drawer_side
    assert len(model.positions) == len(model.orientations) == 2
    assert parent_calls == [(("sentinel",), {"attempt_num": 4})]


@pytest.mark.parametrize(
    "task_class", [kitchen_drawer.OpenDrawer, ChooseMeasuringCup, DrawerCakeSort]
)
def test_no_clear_drawer_side_keeps_parent_robot_pose(monkeypatch, task_class):
    task, model, parent_calls = _task(monkeypatch, task_class, None)

    task._load_model(attempt_num=7)

    np.testing.assert_allclose(model.positions[-1], [0.0, -0.3, 0.793])
    np.testing.assert_allclose(model.orientations[-1], [0.0, 0.0, 0.25])
    np.testing.assert_allclose(model.positions[-1], task.init_robot_base_pos_anchor)
    assert len(model.positions) == len(model.orientations) == 1
    assert parent_calls == [((), {"attempt_num": 7})]
