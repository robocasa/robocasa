from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.utils.object_utils import check_obj_grasped


class _Model:
    def __init__(self, joint_addresses):
        self._joint_addresses = joint_addresses

    def get_joint_qpos_addr(self, joint):
        if joint not in self._joint_addresses:
            raise ValueError(f'No "joint" with name {joint} exists.')
        return self._joint_addresses[joint]


def _env(*, joints, qpos, contact):
    gripper = SimpleNamespace(joints=joints)
    env = SimpleNamespace(
        objects={"sponge": object()},
        robots=[SimpleNamespace(gripper={"right": gripper})],
        sim=SimpleNamespace(
            model=_Model(
                {
                    "gripper0_right_finger_joint1": 0,
                    "gripper0_right_finger_joint2": 1,
                }
            ),
            data=SimpleNamespace(qpos=np.asarray(qpos, dtype=float)),
        ),
    )
    env.check_contact = lambda actual_gripper, actual_obj: (
        actual_gripper is gripper
        and actual_obj is env.objects["sponge"]
        and contact
    )
    return env


@pytest.mark.parametrize(
    ("qpos", "expected"),
    [
        ((0.02, -0.02), True),
        ((0.04, -0.04), False),
    ],
)
def test_check_obj_grasped_preserves_two_finger_closure_check(qpos, expected):
    env = _env(
        joints=(
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ),
        qpos=qpos,
        contact=True,
    )

    assert check_obj_grasped(env, "sponge") is expected


@pytest.mark.parametrize("contact", [False, True])
def test_check_obj_grasped_uses_contact_for_sonic_dex3(contact):
    env = _env(
        joints=(
            "gripper0_right_right_hand_thumb_0_joint",
            "gripper0_right_right_hand_middle_0_joint",
            "gripper0_right_right_hand_index_0_joint",
        ),
        qpos=(0.0, 0.0),
        contact=contact,
    )

    assert check_obj_grasped(env, "sponge", threshold=0.6) is contact
