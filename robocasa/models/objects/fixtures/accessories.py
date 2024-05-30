import numpy as np

from robocasa.models.objects.fixtures.fixture import Fixture
from robosuite.utils.mjcf_utils import string_to_array as s2a, array_to_string as a2s


# For air fryers, etc.
class Accessory(Fixture):
    def __init__(
        self, xml, name,
        pos=None,
        *args, **kwargs
    ):
        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            pos=pos,
            *args,
            **kwargs
        )


class CoffeeMachine(Accessory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turned_on = False

        self._receptacle_pouring_site = self.worldbody.find("./body/body/site[@name='{}{}']".format(
            self.naming_prefix, "receptacle_place_site"
        ))
        self._coffee_liquid_site_names = []
        for postfix in ["coffee_liquid_left", "coffee_liquid_right", "coffee_liquid"]:
            name = "{}{}".format(self.naming_prefix, postfix)
            site = self.worldbody.find("./body/body/site[@name='{}']".format(name))
            if site is not None:
                self._coffee_liquid_site_names.append(name)

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, each region defined as position, x_bounds, y_bounds
        """
        return {
            "bottom": {
                "offset": s2a(self._receptacle_pouring_site.get("pos")),
                "size": (0.01, 0.01),
            }
        }
    
    def get_state(self):
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    def update_state(self, env):
        start_button_pressed = env.check_contact(env.robots[0].gripper["right"], "{}_start_button".format(self.name))

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
        """
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        pour_site_name = "{}{}".format(self.naming_prefix, "receptacle_place_site")
        site_id = env.sim.model.site_name2id(pour_site_name)
        pour_site_pos = env.sim.data.site_xpos[site_id]
        xy_check = np.linalg.norm(obj_pos[0:2] - pour_site_pos[0:2]) < xy_thresh 
        z_check = np.abs(obj_pos[2] - pour_site_pos[2]) < 0.10
        return xy_check and z_check
    
    def gripper_button_far(self, env, th=0.15):
        button_id = env.sim.model.geom_name2id("{}{}".format(self.naming_prefix, "start_button"))
        button_pos = env.sim.data.geom_xpos[button_id]
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        
        gripper_button_far = np.linalg.norm(gripper_site_pos - button_pos) > th
 
        return gripper_button_far
    
    @property
    def nat_lang(self):
        return "coffee machine"


class Toaster(Accessory):

    @property
    def nat_lang(self):
        return "toaster"
    
class Stool(Accessory):
    @property
    def nat_lang(self):
        return "stool"

# For outlets, clocks, paintings, etc.
class WallAccessory(Fixture):
    def __init__(
        self, xml, name,
        pos,
        attach_to=None,
        protrusion=0.02,
        *args, **kwargs
    ):
        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            pos=pos,
            *args,
            **kwargs
        )

        # TODO add in error checking for rotated walls
        # if (pos[1] is None and attach_to is None) or (pos[1] is not None and attach_to is not None):
        #     raise ValueError("Exactly one of y-dimension \"pos\" and \"attach_to\" " \
        #                      "must be specified")
        # if pos[0] is None or pos[2] is None:
        #     raise ValueError("The x and z-dimension position must be specified")

        # the wall to attach accessory to
        self.wall = attach_to
        # how much to protrude out of wall
        if protrusion is not None:
            self.protrusion = protrusion
        else:
            self.protrusion = self.depth / 2

        self._place_accessory()

    def _place_accessory(self):
        if self.wall is None:
            # absolute position was specified
            return
        
        x, y, z = self.pos
        # print(self.wall.wall_side, self.name)

        if self.wall.wall_side == "back":
            y = self.wall.pos[1] - self.protrusion
        elif self.wall.wall_side == "front":
            self.set_euler([0, 0, self.rot + 3.1415])
            y = self.wall.pos[1] + self.protrusion
        elif self.wall.wall_side == "right":
            x = self.wall.pos[0] - self.protrusion
            self.set_euler([0, 0, self.rot - 1.5708])
        elif self.wall.wall_side == "left":
            x = self.wall.pos[0] + self.protrusion
            self.set_euler([0, 0, self.rot + 1.5708])
        elif self.wall.wall_side == "floor":
            raise NotImplementedError()
        else:
            raise ValueError()
        
        self.set_pos([x, y, z])
