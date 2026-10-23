"""Keep live navigation and replay on the same 25 cm success criterion."""

from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.kitchen.atomic.kitchen_navigate import NavigateKitchen


class _NavigationSim:
    def __init__(self, body_name):
        self.model = SimpleNamespace(body_names=[body_name])
        self.model.body_name2id = self.model.body_names.index
        self.data = SimpleNamespace(
            body_xpos=np.zeros((1, 3)),
            body_xmat=np.eye(3).reshape((1, 9)),
        )

    def set_state_from_flattened(self, state):
        _, x, y, z, yaw = state
        self.data.body_xpos[0] = [x, y, z]
        c, s = np.cos(yaw), np.sin(yaw)
        self.data.body_xmat[0] = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        ).reshape(-1)

    def forward(self):
        pass


def _navigation_task(robot_class="SonicG1", body_name="robot0_pelvis"):
    task = object.__new__(NavigateKitchen)
    task.robots = [
        SimpleNamespace(robot_model=type(robot_class, (), {})())
    ]
    task.target_pos = np.zeros(3)
    task.target_ori = np.zeros(3)
    task.sim = _NavigationSim(body_name)
    return task


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ([0.20, 0.0, 0.0], True),
        ([0.2499, 0.0, 0.0], True),
        ([0.25, 0.0, 0.0], True),
        ([0.2501, 0.0, 0.0], False),
        ([0.15, 0.20, 10.0], True),
        ([0.20, 0.20, 0.0], False),
    ],
)
def test_navigation_success_uses_25_cm_horizontal_radius(position, expected):
    task = _navigation_task()
    task.sim.set_state_from_flattened([0.0, *position, 0.0])

    assert bool(task._check_success()) is expected


@pytest.mark.parametrize("yaw", [-0.21, 0.21, np.pi])
def test_navigation_success_still_requires_facing_target(yaw):
    task = _navigation_task()
    task.sim.set_state_from_flattened([0.0, 0.1, 0.0, 0.0, yaw])

    assert not task._check_success()


@pytest.mark.parametrize(
    ("robot_class", "body_name"),
    [
        ("SonicG1", "robot0_pelvis"),
        ("SonicG1", "robot0_base"),
        ("SonicG1Fixed", "robot0_pelvis"),
        ("SonicG1Fixed", "robot0_base"),
        ("OtherMobileRobot", "mobilebase0_base"),
    ],
)
def test_navigation_success_uses_supported_robot_base(robot_class, body_name):
    task = _navigation_task(robot_class, body_name)
    task.sim.set_state_from_flattened([0.0, 0.25, 0.0, 0.0, 0.0])

    assert task._check_success()


def test_navigation_success_rejects_missing_robot_base():
    task = _navigation_task(body_name="unrelated_body")

    assert not task._check_success()
