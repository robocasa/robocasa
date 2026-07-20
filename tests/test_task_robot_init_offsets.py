from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.utils import env_utils as EnvUtils


@pytest.mark.parametrize(
    ("task_name", "expected_offset"),
    [
        ("SlideDishwasherRack", (-0.50, 0.0)),
        ("PickPlaceDrawerToCounter", (0.0, -0.35)),
        ("OpenDrawer", (0.0, 0.0)),
    ],
)
def test_task_robot_init_offsets_use_two_task_overrides_and_a_zero_default(
    task_name, expected_offset
):
    assert EnvUtils.get_task_robot_init_offset(task_name) == expected_offset


def test_task_robot_offsets_apply_only_to_sonic_g1():
    class SonicG1:
        pass

    class PandaOmron:
        pass

    task_offset = (-0.50, 0.0)
    sonic_env = SimpleNamespace(
        robots=[SimpleNamespace(robot_model=SonicG1())],
        robot_init_offset=task_offset,
    )
    panda_env = SimpleNamespace(
        robots=[SimpleNamespace(robot_model=PandaOmron())],
        robot_init_offset=task_offset,
    )

    assert EnvUtils.get_effective_robot_init_offset(sonic_env) == task_offset
    assert EnvUtils.get_effective_robot_init_offset(panda_env) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("yaw", "offset"),
    [
        (0.0, (-0.50, 0.0)),
        (np.pi / 2, (-0.50, 0.0)),
        (np.pi, (0.0, -0.35)),
        (-np.pi / 2, (0.0, -0.35)),
    ],
)
def test_apply_robot_init_offset_uses_robot_local_lateral_and_longitudinal_axes(
    yaw, offset
):
    position = np.array([1.0, -2.0, 0.793])
    orientation = np.array([0.0, 0.0, yaw])
    lateral, longitudinal = offset
    expected = position + np.array(
        [
            np.sin(yaw) * lateral + np.cos(yaw) * longitudinal,
            -np.cos(yaw) * lateral + np.sin(yaw) * longitudinal,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        EnvUtils.apply_robot_init_offset(position, orientation, offset),
        expected,
        atol=1e-12,
    )
