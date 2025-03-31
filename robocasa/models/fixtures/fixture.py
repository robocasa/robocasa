import abc
import os
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from enum import IntEnum

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    find_parent,
    string_to_array,
    xml_path_completion,
)

import robocasa
import robocasa.macros as macros
from robocasa.models.objects.objects import MujocoXMLObjectRobocasa
from robocasa.utils.object_utils import get_pos_after_rel_offset


def get_texture_name_from_file(file):
    """
    Extract texture name from filename.
    eg: ../robosuite/models/assets/textures/flat/gray.png -> flat/gray
    """
    suffix_path = file.split("textures/")[1]
    name = suffix_path.split(".")[0]
    return name


class FixtureType(IntEnum):
    """
    Enum for fixture types in robosuite kitchen environments.
    """

    COUNTER = 1
    MICROWAVE = 2
    STOVE = 3
    SINK = 4
    CABINET = 5
    DRAWER = 6
    SHELF = 7
    COFFEE_MACHINE = 8
    DOOR = 9
    DOOR_HINGE = 10
    DOOR_HINGE_SINGLE = 11
    DOOR_HINGE_DOUBLE = 12
    DOOR_TOP_HINGE = 13
    DOOR_TOP_HINGE_SINGLE = 14
    DOOR_TOP_HINGE_DOUBLE = 15
    CABINET_TOP = 16
    TOASTER = 17
    DINING_COUNTER = 18
    TOP_DRAWER = 19
    STOOL = 20
    ISLAND = 21
    COUNTER_NON_CORNER = 22
    FRIDGE = 23
    DISHWASHER = 24
    OVEN = 25
    TOASTER_OVEN = 26
    BLENDER = 27
    STAND_MIXER = 28
    ELECTRIC_KETTLE = 29


class Fixture(MujocoXMLObjectRobocasa):
    """
    Base class for fixtures in robosuite kitchen environments.

    Args:
        xml (str): Path to the MJCF xml file to load the fixture from

        name (str): Name of the fixture

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a visual geom copy

        pos (3-tuple): (x, y, z) position of the fixture

        scale (float): 3D Scaling factor for the fixture

        size (3-tuple): desired (width, depth, height) of the fixture
    """

    RESET_REGION_NAMES = ["int"]

    def __init__(
        self,
        xml,
        name,
        duplicate_collision_geoms=True,
        pos=None,
        scale=1,
        size=None,
        placement=None,
        rng=None,
    ):
        if not xml.endswith(".xml"):
            xml = os.path.join(xml, "model.xml")

        super().__init__(
            xml_path_completion(xml, root=robocasa.models.assets_root),
            name=name,
            joints=None,
            duplicate_collision_geoms=duplicate_collision_geoms,
            scale=scale,
        )
        if pos is not None:
            self.set_pos(pos)

        # set up exterior and interior sites
        self._regions = dict()

        # search for all geom regions
        geom_list = find_elements(
            self.worldbody,
            tags="geom",
            return_first=False,
        )

        if geom_list is None:
            geom_list = []
        for geom in geom_list:
            g_name = geom.get("name")
            if g_name is None:
                continue
            g_name = g_name.split(self.naming_prefix)[
                1
            ]  # strip out fixture name from prefix

            if not g_name.startswith("reg_"):
                continue

            rgba = string_to_array(geom.get("rgba"))
            # if macros.SHOW_SITES:
            #     rgba[0:3] = np.random.uniform(0, 1, (3,))
            #     rgba[-1] = 1.0
            # else:
            #     rgba[-1] = 0.0
            # geom.set("rgba", array_to_string(rgba))

            reg_dict = dict()
            reg_pos = string_to_array(geom.get("pos"))
            reg_halfsize = string_to_array(geom.get("size"))
            p0 = reg_pos + [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            px = reg_pos + [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            py = reg_pos + [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
            pz = reg_pos + [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]

            reg_dict["elem"] = geom
            reg_dict["p0"] = p0
            reg_dict["px"] = px
            reg_dict["py"] = py
            reg_dict["pz"] = pz
            prefix = g_name[4:]
            self._regions[prefix] = reg_dict

        # scale based on specified max dimension
        if size is not None:
            self.set_scale_from_size(size)

        # based on exterior points, overwritten by subclasses (e.g. Counter) that do not have such sites
        self.size = np.array([self.width, self.depth, self.height])

        # set offset between center of object and center of exterior bounding boxes
        if self.width is not None:
            try:
                # calculate based on bounding points
                p0 = self._regions["main"]["p0"]
                px = self._regions["main"]["px"]
                py = self._regions["main"]["py"]
                pz = self._regions["main"]["pz"]
                self.origin_offset = np.array(
                    [
                        np.mean((p0[0], px[0])),
                        np.mean((p0[1], py[1])),
                        np.mean((p0[2], pz[2])),
                    ]
                )
            except KeyError:
                self.origin_offset = [0, 0, 0]
        else:
            self.origin_offset = [0, 0, 0]
        self.origin_offset = np.array(self.origin_offset)

        # placement config, for determining where to place fixture (most fixture will not use this)
        self._placement = placement

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def set_origin(self, origin):
        """
        Set the origin of the fixture to a specified position

        Args:
            origin (3-tuple): new (x, y, z) position of the fixture
        """
        # compute new position
        fixture_rot = np.array([0, 0, self.rot])
        fixture_mat = T.euler2mat(fixture_rot)

        pos = origin + np.dot(fixture_mat, -self.origin_offset)
        self.set_pos(pos)

    def set_scale_from_size(self, size):
        """
        Set the scale of the fixture based on the desired size. If any of the dimensions are None,
        the scaling factor will be the same as one of the other two dimensions

        Args:
            size (3-tuple): (width, depth, height) of the fixture
        """
        # check that the argument is valid
        assert len(size) == 3

        # calculate and set scale according to specification
        scale = [None, None, None]
        cur_size = [self.width, self.depth, self.height]
        for (i, t) in enumerate(size):
            if t is not None:
                scale[i] = t / cur_size[i]

        scale[0] = scale[0] or scale[2] or scale[1]
        scale[1] = scale[1] or scale[0] or scale[2]
        scale[2] = scale[2] or scale[0] or scale[1]
        self.set_scale(scale)

        for (reg_name, reg_dict) in self._regions.items():
            for (k, v) in reg_dict.items():
                if isinstance(v, np.ndarray):
                    reg_dict[k] = v * scale

    def get_reset_regions(self, env, reset_region_names=None):
        """
        returns dictionary of reset regions, each region defined as position, x_bounds, y_bounds
        """
        reset_regions = {}
        if reset_region_names is None:
            reset_region_names = self.RESET_REGION_NAMES
        for reg_name in reset_region_names:
            reg_dict = self._regions.get(reg_name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]
            pz = reg_dict["pz"]
            reset_regions[reg_name] = {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        return reset_regions

    def sample_reset_region(self, *args, **kwargs):
        regions = self.get_reset_regions(*args, **kwargs)
        return self.rng.choice(list(regions.values()))

    def get_site_info(self, sim):
        """
        returns user defined sites (eg. the table top, door handle sites, handle sites, shelf sites, etc)
        requires sim as position of sites can change during simulation.
        """
        info = {}
        for s in self._sites:
            name = "{}{}".format(self.naming_prefix, s)
            info[name] = sim.data.get_site_xpos(name)
        return info

    @abc.abstractmethod
    def get_state(self):
        """
        get the current state of the fixture
        """
        return

    @abc.abstractmethod
    def update_state(self, env):
        """
        update internal state of fixture
        """
        return

    @property
    def pos(self):
        return string_to_array(self._obj.get("pos"))

    @property
    def quat(self):
        quat = self._obj.get("quat")
        if quat is None:
            # no rotation applied
            quat = "0 0 0 0"
        return string_to_array(quat)

    @property
    def euler(self):
        euler = self._obj.get("euler")
        if euler is None:
            # no rotation applied
            euler = "0 0 0"
        return np.array(string_to_array(euler))

    @property
    def horizontal_radius(self):
        """
        override the default behavior of only looking at first dimension for radius
        """
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = string_to_array(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    @property
    def bottom_offset(self):
        return self._regions["main"]["p0"]

    @property
    def width(self):
        """
        for getting the width of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main" in self._regions:
            main_p0 = self._regions["main"]["p0"]
            main_px = self._regions["main"]["px"]
            w = main_px[0] - main_p0[0]
            return w
        else:
            return None

    @property
    def depth(self):
        """
        for getting the depth of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main" in self._regions:
            main_p0 = self._regions["main"]["p0"]
            main_py = self._regions["main"]["py"]
            d = main_py[1] - main_p0[1]
            return d
        else:
            return None

    @property
    def height(self):
        """
        for getting the height of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main" in self._regions:
            main_p0 = self._regions["main"]["p0"]
            main_pz = self._regions["main"]["pz"]
            h = main_pz[2] - main_p0[2]
            return h
        else:
            return None

    def set_regions(self, region_dict):
        """
        Set the positions of the exterior and interior bounding box sites of the object

        Args:
            region_dict (dict): Dictionary of regions (containing pos, halfsize)
        """
        for (name, reg) in region_dict.items():
            pos = np.array(reg["pos"])
            halfsize = np.array(reg["halfsize"])
            self._regions[name]["elem"].set("pos", array_to_string(pos))
            self._regions[name]["elem"].set("size", array_to_string(halfsize))

            # compute boundary points for reference
            p0 = pos + np.array([-halfsize[0], -halfsize[1], -halfsize[2]])
            px = pos + np.array([halfsize[0], -halfsize[1], -halfsize[2]])
            py = pos + np.array([-halfsize[0], halfsize[1], -halfsize[2]])
            pz = pos + np.array([-halfsize[0], -halfsize[1], halfsize[2]])
            self._regions[name]["p0"] = p0
            self._regions[name]["px"] = px
            self._regions[name]["py"] = py
            self._regions[name]["pz"] = pz

    def get_ext_sites(self, all_points=False, relative=True):
        """
        Get the exterior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        sites = [
            self._regions["main"]["p0"],
            self._regions["main"]["px"],
            self._regions["main"]["py"],
            self._regions["main"]["pz"],
        ]

        if all_points:
            p0, px, py, pz = sites
            sites += [
                np.array([p0[0], py[1], pz[2]]),
                np.array([px[0], py[1], pz[2]]),
                np.array([px[0], py[1], p0[2]]),
                np.array([px[0], p0[1], pz[2]]),
            ]

        if relative is False:
            sites = [get_pos_after_rel_offset(self, offset) for offset in sites]

        return sites

    def get_int_sites(self, all_points=False, relative=True):
        """
        Get the interior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            dict: a dictionary of interior areas, each with 4 or 8 points
        """
        sites_dict = {}
        for prefix in self.RESET_REGION_NAMES:
            reg_dict = self._regions.get(prefix, None)
            if reg_dict is None:
                continue

            sites = [
                reg_dict["p0"],
                reg_dict["px"],
                reg_dict["py"],
                reg_dict["pz"],
            ]

            if all_points:
                p0, px, py, pz = sites
                sites += [
                    np.array([p0[0], py[1], pz[2]]),
                    np.array([px[0], py[1], pz[2]]),
                    np.array([px[0], py[1], p0[2]]),
                    np.array([px[0], p0[1], pz[2]]),
                ]

            if relative is False:
                sites = [get_pos_after_rel_offset(self, offset) for offset in sites]

            sites_dict[prefix] = sites

        return sites_dict

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full set of bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = self.get_ext_sites(all_points=True, relative=True)

        if trans is None:
            trans = self.pos
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.array([0, 0, self.rot])
            rot = T.euler2mat(rot)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    def _remove_element(self, elem):
        # # This method not currently working
        # parent_elem = find_parent(self._obj, elem)
        # parent_elem.remove(elem)

        # TODO: actually delete element?
        # Below might actually be faster than searching for/deleting element
        elem.set("rgba", "0 0 0 0")
        elem.set("pos", "0 0 10")
        elem.set("size", "0.01 0.01 0.01")

    @property
    def nat_lang(self):
        return self.name


class ProcGenFixture(Fixture):
    def exclude_from_prefixing(self, inp):
        """
        Exclude all shared materials and their associated names from being prefixed.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        if "tex" in inp:
            return True

        if isinstance(inp, ET.Element):
            return inp.tag in ["texture"]

        return False
