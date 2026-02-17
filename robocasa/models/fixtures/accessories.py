import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a

from robocasa.models.fixtures import Fixture


class Accessory(Fixture):
    """
    Base class for all accessories/Miscellaneous objects

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object

        pos (list): position of the object
    """

    def __init__(
        self,
        xml,
        name,
        has_free_joints=False,
        pos=None,
        is_clutter=False,
        *args,
        **kwargs
    ):
        if has_free_joints:
            joints = [dict(type="free", damping="0.0005")]
        else:
            joints = None
        self.is_clutter = is_clutter

        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            joints=joints,
            pos=pos,
            *args,
            **kwargs
        )


# For outlets, clocks, paintings, etc.
class WallAccessory(Fixture):
    """
    Class for wall accessories. These are objects that are attached to walls, such as outlets, clocks, paintings, etc.

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object

        pos (list): position of the object

        attach_to (Wall): The wall to attach the object to

        protrusion (float): How much to protrude out of the wall when placing the object
    """

    def __init__(
        self, xml, name, pos, attach_to=None, protrusion=0.02, *args, **kwargs
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
        if self.wall is not None:
            if protrusion is not None:
                self.protrusion = protrusion
            else:
                self.protrusion = (
                    self.height / 2
                    if self.wall.wall_side == "floor"
                    else self.depth / 2
                )
                self.protrusion += self.wall.size[2] if self.wall is not None else 0
        else:
            self.protrusion = None

        self._place_accessory()

    def _place_accessory(self):
        """
        Place the accessory on the wall
        """
        if self.wall is None:
            # absolute position was specified
            return

        x, y, z = self.pos
        # print(self.wall.wall_side, self.name)

        # update position and rotation of the object based on the wall it attaches to
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
            z = self.wall.pos[2] + self.protrusion
        else:
            raise ValueError()

        self.set_pos([x, y, z])


class Stool(WallAccessory):
    def __init__(
        self,
        xml,
        name,
        pos,
        attach_to=None,
        protrusion=None,
        z_rot=None,
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml,
            name=name,
            pos=pos,
            attach_to=attach_to,
            protrusion=protrusion,
            *args,
            **kwargs
        )
        if z_rot is not None:
            self.set_euler([0, 0, z_rot])

    @property
    def nat_lang(self):
        return "stool"
