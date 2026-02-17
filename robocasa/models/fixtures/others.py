import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a
from robosuite.utils.mjcf_utils import xml_path_completion
import robosuite.utils.transform_utils as T

import robocasa
from robocasa.models.fixtures import Fixture


class Box(BoxObject):
    """
    Initializes a box object. Mainly used for filling in gaps in the environment like in corners or beneath bottom cabinets

    Args:
        pos (list): position of the object

        size (list): size of the object

        name (str): name of the object

        texture (str): path to texture file

        mat_attrib (dict): material attributes

        tex_attrib (dict): texture attributes
    """

    def __init__(
        self,
        pos,
        size,
        name="box",
        texture="textures/wood/dark_wood_parquet.png",
        mat_attrib={"shininess": "0.1"},
        tex_attrib={"type": "cube"},
        rng=None,
        *args,
        **kwargs
    ):
        texture = xml_path_completion(texture, root=robocasa.models.assets_root)
        material = CustomMaterial(
            texture=texture,
            tex_name="box",
            mat_name="box_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        super().__init__(
            name=name,
            material=material,
            joints=None,
            # divide by 2 per mujoco convention
            size=[x / 2 for x in size],
        )

        self.size = size
        self.pos = pos
        if pos is not None:
            self._obj.set("pos", a2s(pos))

        # for relative positioning
        self.origin_offset = np.array([0, 0, 0])
        self.scale = 1

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def set_pos(self, pos):
        """
        Set the position of the object

        Args:
            pos (list): position of the object
        """
        self.pos = pos
        self._obj.set("pos", a2s(pos))

    def update_state(self, env):
        pass

    @property
    def rot(self):
        """
        Returns the rotation of the object only on the z-axis

        Returns:
            float: rotation
        """
        rot = s2a(self._obj.get("euler", "0.0 0.0 0.0"))
        return rot[2]


class Wall(BoxObject):
    """
    Initializes a wall object. Used for creating walls in the environment

    Args:
        name (str): name of the object

        texture (str): path to texture file

        pos (list): position of the object

        quat (list): quaternion of the object

        size (list): size of the object

        wall_side (str): which side the wall is on (back, front, left, right, floor)

        mat_attrib (dict): material attributes

        tex_attrib (dict): texture attributes

        backing (bool): whether this is a backing wall

        backing_extended (list): whether the backing is extended on the left and right

        default_wall_th (float): default thickness of the wall

        default_backing_th (float): default thickness of the backing
    """

    def __init__(
        self,
        name="wall",
        texture="textures/bricks/white_bricks.png",
        pos=None,
        quat=None,
        size=None,
        wall_side="back",
        mat_attrib={
            "texrepeat": "3 3",
            "reflectance": "0.1",
            "shininess": "0.1",
            "texuniform": "true",
        },
        tex_attrib={"type": "2d"},
        # parameters used for alignment
        backing=False,
        backing_extended=[False, False],
        default_wall_th=0.02,
        default_backing_th=0.1,
        enclosing_wall=False,
        rng=None,
        *args,
        **kwargs
    ):
        # change texture if used for backing
        if backing:
            texture = "textures/flat/light_gray.png"
        texture = xml_path_completion(texture, root=robocasa.models.assets_root)
        material = CustomMaterial(
            texture=texture,
            tex_name="wall",
            mat_name="wall_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.wall_side = wall_side
        self.enclosing_wall = enclosing_wall
        # set the rotation according to which side the wall is on
        if self.wall_side is not None:
            quat = self.get_quat()

        # align everything to account for thickness & backing
        if self.wall_side == "floor":
            size[0] += default_wall_th * 2
            size[1] += default_wall_th * 2
            pos[2] -= size[2]
            if backing:
                pos[2] -= default_wall_th * 2
        else:
            size[0] += default_wall_th * 2
            shift = size[2] if not backing else size[2] + default_wall_th * 2
            if self.wall_side == "left":
                pos[0] -= shift
            elif self.wall_side == "right":
                pos[0] += shift
            elif self.wall_side == "back":
                pos[1] += shift
            elif self.wall_side == "front":
                pos[1] -= shift

            if backing:
                size[1] += default_wall_th + default_backing_th
                pos[2] -= default_wall_th + default_backing_th

                # extend left/right side to form a perfect box
                if backing_extended[0]:
                    size[0] += default_backing_th
                    if self.wall_side in ["left", "right"]:
                        pos[1] += default_backing_th
                    else:
                        pos[0] -= default_backing_th
                if backing_extended[1]:
                    size[0] += default_backing_th
                    if self.wall_side in ["left", "right"]:
                        pos[1] -= default_backing_th
                    else:
                        pos[0] += default_backing_th

        super().__init__(
            name=name, material=material, joints=None, size=size, *args, **kwargs
        )

        self.pos = pos
        if pos is not None:
            self._obj.set("pos", a2s(pos))
        if quat is not None:
            self._obj.set("quat", a2s(quat))

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def set_pos(self, pos):
        """
        Set the position of the object

        Args:
            pos (list): position of the object
        """
        self.pos = pos
        self._obj.set("pos", a2s(pos))

    def get_quat(self):
        """
        Returns the quaternion of the object based on the wall side

        Returns:
            list: quaternion
        """
        side_rots = {
            "back": [-0.707, 0.707, 0, 0],
            "front": [0, 0, 0.707, -0.707],
            "left": [0.5, 0.5, -0.5, -0.5],
            "right": [0.5, -0.5, -0.5, 0.5],
            "floor": [0.707, 0, 0, 0.707],
        }
        if self.wall_side not in side_rots:
            raise ValueError()
        return side_rots[self.wall_side]

    def update_state(self, env):
        pass

    def _get_reordered_bbox_pts(self, pts):
        """
        Reorder the points of the bounding box to be in a specific order
        This is because after a rotation px may not represent an offset in the x direction anymore.
        pz may not represent an offset in the z direction anymore, etc.
        """
        offs = [p - self.pos for p in pts]
        # for each corner, build its “signature” (+1 or -1 on each axis)
        signs = [tuple(int(np.sign(o[i])) for i in range(3)) for o in offs]

        # map signatures → index in the final list
        sig_to_idx = {
            (-1, -1, -1): 0,  # p0
            (1, -1, -1): 1,  # px
            (-1, 1, -1): 2,  # py
            (-1, -1, 1): 3,  # pz
            (1, 1, -1): 4,  # pxy
            (1, -1, 1): 5,  # pxz
            (-1, 1, 1): 6,  # pyz
            (1, 1, 1): 7,  # pxyz
        }

        # now build the reordered list
        out = [None] * len(pts)
        for p, s in zip(pts, signs):
            out[sig_to_idx[s]] = p

        return out

    def get_ext_sites(self, all_points=False, relative=True):
        """
        Get the exterior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        reg_halfsize = self.get_bounding_box_half_size()
        p0 = [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
        px = [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
        py = [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
        pz = [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]
        sites = [
            p0,
            px,
            py,
            pz,
        ]

        p0, px, py, pz = sites
        sites += [
            np.array([p0[0], py[1], pz[2]]),
            np.array([px[0], py[1], pz[2]]),
            np.array([px[0], py[1], p0[2]]),
            np.array([px[0], p0[1], pz[2]]),
        ]

        quat = np.array(self.get_quat())
        if relative is False:
            sites = [
                self._get_pos_after_rel_tranformation(offset, quat) for offset in sites
            ]
            sites = self._get_reordered_bbox_pts(sites)

        if not all_points:
            return sites[:4]
        return sites

    def _get_pos_after_rel_tranformation(self, offset, quat):
        fixture_mat = T.quat2mat(T.convert_quat(quat))
        return self.pos + np.dot(fixture_mat, offset)


class Floor(Wall):
    def __init__(
        self,
        size,
        name="wall",
        texture="textures/bricks/red_bricks.png",
        mat_attrib={
            "texrepeat": "2 2",
            "texuniform": "true",
            "reflectance": "0.1",
            "shininess": "0.1",
        },
        *args,
        **kwargs
    ):
        # swap x, y axes due to rotation
        size = [size[1], size[0], size[2]]

        texture = xml_path_completion(texture, root=robocasa.models.assets_root)

        # everything is the same except the plane is rotated to be horizontal
        super().__init__(
            name,
            texture,
            # horizontal plane
            wall_side="floor",
            size=size,
            mat_attrib=mat_attrib,
            *args,
            **kwargs
        )
