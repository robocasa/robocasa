from robocasa.models.fixtures import Fixture
import numpy as np
from robosuite.utils.mjcf_utils import string_to_array


class CoffeeMachine(Fixture):
    """
    Coffee machine fixture class
    """

    def __init__(self, xml, name="coffee_machine", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

        self._turned_on = False
        # site where coffee liquid is poured
        self._receptacle_pouring_site = self.worldbody.find(
            "./body/body/site[@name='{}{}']".format(
                self.naming_prefix, "receptacle_place_site"
            )
        )

        # sites which act as the simulated coffee liquid which pours out when the start button is pressed
        self._coffee_liquid_site_names = []
        for postfix in ["coffee_liquid_left", "coffee_liquid_right", "coffee_liquid"]:
            name = "{}{}".format(self.naming_prefix, postfix)
            site = self.worldbody.find("./body/body/site[@name='{}']".format(name))
            if site is not None:
                self._coffee_liquid_site_names.append(name)

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, usually used when initialzing a mug under the coffee machine
        """
        return {
            "bottom": {
                "offset": string_to_array(self._receptacle_pouring_site.get("pos")),
                "size": (0.01, 0.01),
            }
        }

    def get_state(self):
        """
        returns whether the coffee machine is turned on or off as a dictionary with the turned_on key
        """
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    def update_state(self, env):
        """
        Checks if the gripper is pressing the start button. If this is the first time the gripper pressed the button,
        the coffee machine is turned on, and the coffee liquid sites are turned on.

        Args:
            env (MujocoEnv): The environment to check the state of the coffee machine in
        """
        start_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_start_button".format(self.name)
        )

        if self._turned_on is False and start_button_pressed:
            self._turned_on = True

        for site_name in self._coffee_liquid_site_names:
            site_id = env.sim.model.site_name2id(site_name)
            if self._turned_on:
                env.sim.model.site_rgba[site_id][3] = 1.0
            else:
                env.sim.model.site_rgba[site_id][3] = 0.0

    def check_receptacle_placement_for_pouring(self, env, obj_name, xy_thresh=0.04):
        """
        check whether receptacle is placed under coffee machine for pouring

        Args:
            env (MujocoEnv): The environment to check the state of the coffee machine in

            obj_name (str): name of the object

            xy_thresh (float): threshold for xy distance between object and receptacle

        Returns:
            bool: True if object is placed under coffee machine, False otherwise
        """
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        pour_site_name = "{}{}".format(self.naming_prefix, "receptacle_place_site")
        site_id = env.sim.model.site_name2id(pour_site_name)
        pour_site_pos = env.sim.data.site_xpos[site_id]
        xy_check = np.linalg.norm(obj_pos[0:2] - pour_site_pos[0:2]) < xy_thresh
        z_check = np.abs(obj_pos[2] - pour_site_pos[2]) < 0.10
        return xy_check and z_check

    def gripper_button_far(self, env, th=0.15):
        """
        check whether gripper is far from the start button

        Args:
            env (MujocoEnv): The environment to check the state of the coffee machine in

            th (float): threshold for distance between gripper and button

        Returns:
            bool: True if gripper is far from the button, False otherwise
        """
        button_id = env.sim.model.geom_name2id(
            "{}{}".format(self.naming_prefix, "start_button")
        )
        button_pos = env.sim.data.geom_xpos[button_id]
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]

        gripper_button_far = np.linalg.norm(gripper_site_pos - button_pos) > th

        return gripper_button_far

    @property
    def nat_lang(self):
        return "coffee machine"
