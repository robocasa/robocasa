import numpy as np

import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import Fixture


class Microwave(Fixture):
    """
    Microwave fixture class. Supports turning on and off the microwave, and opening and closing the door

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    def __init__(
        self,
        xml="fixtures/microwaves/standard",
        name="microwave",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._turned_on = False

    def is_open(self, env, th=0.90):
        """
        checks whether the fixture is open
        """
        return super().is_open(env, self.door_joint_names, th=th)

    def is_closed(self, env, th=0.005):
        """
        checks whether the fixture is closed
        """
        return super().is_closed(env, self.door_joint_names, th=th)

    def get_state(self):
        """
        Returns:
            dict: maps turned_on to whether the microwave is turned on
        """
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    @property
    def door_joint_names(self):
        return [f"{self.name}_microjoint"]

    @property
    def handle_name(self):
        return "{}_door_handle".format(self.name)

    @property
    def door_name(self):
        return "{}_door".format(self.name)

    def update_state(self, env):
        """
        If the microwave is open, the state is set to off. Otherwise, if the gripper
        is pressing the start button, the microwave will stay/turn on. If the gripper is
        pressing the stop button, the microwave will stay/turn off.

        Args:
            env (MujocoEnv): The environment to check the state of the microwave in

        """
        start_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_start_button".format(self.name)
        )
        stop_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_stop_button".format(self.name)
        )

        door_open = self.is_open(env)
        if door_open:
            self._turned_on = False
        else:
            if self._turned_on is True and stop_button_pressed:
                self._turned_on = False
            elif self._turned_on is False and start_button_pressed:
                self._turned_on = True

    def gripper_button_far(self, env, button, th=0.15):
        """
        check whether gripper is far from the start button

        Args:
            env (MujocoEnv): The environment to check the state of the microwave in

            button (str): button to check

            th (float): threshold for distance between gripper and button

        Returns:
            bool: True if gripper is far from the button, False otherwise
        """
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
