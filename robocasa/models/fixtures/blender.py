from robocasa.models.fixtures import Fixture
import numpy as np
from robocasa.utils import object_utils as OU


class Blender(Fixture):
    """
    Blender fixture class
    """

    _BLENDER_LID_POS_THRESH = 0.04

    def __init__(self, xml, name="blender", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self.lid_closed_pos = None
        self._lid_on_blender = True
        self._turned_on = False
        self._button_contact_prev_timestep = False
        self.blender_lid = None

    def add_auxiliary_fixture(self, auxiliary_fixture):
        self.blender_lid = auxiliary_fixture

    def get_lid_closed_pos(self, env):
        if self.lid_closed_pos is None:
            self.lid_closed_pos = OU.get_pos_after_rel_offset(self, self.anchor_offset)
        return self.lid_closed_pos

    def get_curr_lid_pos(self, env):
        if self.blender_lid is None:
            return None
        return env.sim.data.get_body_xpos(f"{self.blender_lid.name}_main")

    def update_state(self, env):
        curr_lid_pos = self.get_curr_lid_pos(env)
        if curr_lid_pos is None:
            self._lid_on_blender = False
        else:
            closed_lid_pos = self.get_lid_closed_pos(env)
            self._lid_on_blender = (
                np.linalg.norm(curr_lid_pos - closed_lid_pos)
                < self._BLENDER_LID_POS_THRESH
            ) and OU.check_fxtr_upright(env, f"{self.blender_lid.name}_main", th=7)

        button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_power_button_main".format(self.name)
        )
        # since the state updates very often and the same button is used for turning on/off
        # we look at the release of the button to determine the state. If we look at the press then
        # the state will flicker between on and off
        switch_state = self._button_contact_prev_timestep and not button_pressed

        if not self._lid_on_blender:
            self._turned_on = False
        else:
            if self._turned_on is True and switch_state:
                self._turned_on = False
            elif self._turned_on is False and switch_state:
                self._turned_on = True
        self._button_contact_prev_timestep = button_pressed

    def get_state(self):
        state = dict(
            lid_on_blender=self._lid_on_blender,
            turned_on=self._turned_on,
        )
        return state

    @property
    def nat_lang(self):
        return "blender"


class BlenderLid(Fixture):
    def __init__(self, xml, name="blender_lid", has_free_joints=True, *args, **kwargs):
        if has_free_joints:
            joints = [dict(type="free", damping="0.0005")]
        else:
            joints = None

        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            joints=joints,
            *args,
            **kwargs,
        )

    @property
    def nat_lang(self):
        return "blender lid"
