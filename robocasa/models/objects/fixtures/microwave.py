import numpy as np

import robocasa.utils.object_utils as OU
from robocasa.models.objects.fixtures.fixture import Fixture


class Microwave(Fixture):
    def __init__(
        self,
        xml="fixtures/microwaves/orig_microwave",
        name="microwave",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._turned_on = False

    def set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = np.pi / 2

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        sign = -1

        env.sim.data.set_joint_qpos(
            "{}_microjoint".format(self.name),
            sign * rng.uniform(desired_min, desired_max),
        )

    def get_door_state(self, env):
        sim = env.sim
        hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_microjoint")]
        hinge_qpos = -hinge_qpos  # negate as micro joints are left door hinges

        # convert to percentages
        door = OU.normalize_joint_value(hinge_qpos, joint_min=0, joint_max=np.pi / 2)

        return {
            "door": door,
        }

    def get_state(self):
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    @property
    def handle_name(self):
        return "{}_door_handle".format(self.name)

    @property
    def door_name(self):
        return "{}_door".format(self.name)

    def update_state(self, env):
        start_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_start_button".format(self.name)
        )
        stop_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_stop_button".format(self.name)
        )

        door_state = self.get_door_state(env)["door"]
        door_open = door_state > 0.005

        if door_open:
            self._turned_on = False
        else:
            if self._turned_on is True and stop_button_pressed:
                self._turned_on = False
            elif self._turned_on is False and start_button_pressed:
                self._turned_on = True

    def gripper_button_far(self, env, button, th=0.15):
        assert button in ["start_button", "stop_button"]
        button_id = env.sim.model.geom_name2id(
            "{}{}".format(self.naming_prefix, button)
        )
        button_pos = env.sim.data.geom_xpos[button_id]
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]

        gripper_button_far = np.linalg.norm(gripper_site_pos - button_pos) > th

        return gripper_button_far

    @property
    def nat_lang(self):
        return "microwave"
