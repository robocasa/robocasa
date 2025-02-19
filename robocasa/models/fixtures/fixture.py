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
from robocasa.models.objects import MujocoXMLObject
from robocasa.utils.object_utils import get_pos_after_rel_offset


def site_pos(site):
    if isinstance(site, np.ndarray):
        return site
    return string_to_array(site.get("pos"))


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


class Fixture(MujocoXMLObject):
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
        self._bounds_sites = dict()

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
            if macros.SHOW_SITES:
                rgba[-1] = 0.25
            else:
                rgba[-1] = 0.0
            geom.set("rgba", array_to_string(rgba))

            reg_pos = string_to_array(geom.get("pos"))
            reg_size = string_to_array(geom.get("size"))
            # ## special case: if
            # if np.all(reg_size <= 0.0001):
            #     reg_size = [0.0, 0.0, 0.0]
            p0 = reg_pos + [-reg_size[0], -reg_size[1], -reg_size[2]]
            px = reg_pos + [reg_size[0], -reg_size[1], -reg_size[2]]
            py = reg_pos + [-reg_size[0], reg_size[1], -reg_size[2]]
            pz = reg_pos + [-reg_size[0], -reg_size[1], reg_size[2]]
            prefix = g_name[4:]

            self._bounds_sites[prefix + "_p0"] = p0
            self._bounds_sites[prefix + "_px"] = px
            self._bounds_sites[prefix + "_py"] = py
            self._bounds_sites[prefix + "_pz"] = pz

        # scale based on specified max dimension
        if size is not None:
            self.set_scale_from_size(size)

        # based on exterior points, overwritten by subclasses (e.g. Counter) that do not have such sites
        self.size = np.array([self.width, self.depth, self.height])

        # set offset between center of object and center of exterior bounding boxes
        if self.width is not None:
            try:
                # calculate based on bounding points
                p0 = site_pos(self._bounds_sites["main_body_p0"])
                px = site_pos(self._bounds_sites["main_body_px"])
                py = site_pos(self._bounds_sites["main_body_py"])
                pz = site_pos(self._bounds_sites["main_body_pz"])
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

        for (k, v) in self._bounds_sites.items():
            if isinstance(v, np.ndarray):
                self._bounds_sites[k] = v * scale

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, each region defined as position, x_bounds, y_bounds
        """
        reset_regions = {}
        for reg_name in self.RESET_REGION_NAMES:
            p0 = self._bounds_sites.get(f"{reg_name}_p0", None)
            if p0 is None:
                continue
            px = self._bounds_sites[f"{reg_name}_px"]
            py = self._bounds_sites[f"{reg_name}_py"]
            pz = self._bounds_sites[f"{reg_name}_pz"]
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
        return site_pos(self._bounds_sites["main_body_p0"])

    @property
    def width(self):
        """
        for getting the width of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main_body_px" in self._bounds_sites:
            main_body_p0 = site_pos(self._bounds_sites["main_body_p0"])
            main_body_px = site_pos(self._bounds_sites["main_body_px"])
            w = main_body_px[0] - main_body_p0[0]
            return w
        else:
            return None

    @property
    def depth(self):
        """
        for getting the depth of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main_body_py" in self._bounds_sites:
            main_body_p0 = site_pos(self._bounds_sites["main_body_p0"])
            main_body_py = site_pos(self._bounds_sites["main_body_py"])
            d = main_body_py[1] - main_body_p0[1]
            return d
        else:
            return None

    @property
    def height(self):
        """
        for getting the height of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "main_body_pz" in self._bounds_sites:
            main_body_p0 = site_pos(self._bounds_sites["main_body_p0"])
            main_body_pz = site_pos(self._bounds_sites["main_body_pz"])
            h = main_body_pz[2] - main_body_p0[2]
            return h
        else:
            return None

    def set_bounds_sites(self, pos_dict):
        """
        Set the positions of the exterior and interior bounding box sites of the object

        Args:
            pos_dict (dict): Dictionary of sites and their new positions
        """
        for (name, pos) in pos_dict.items():
            # self._bounds_sites[name].set("pos", array_to_string(pos))
            self._bounds_sites[name] = np.array(pos)

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
            site_pos(self._bounds_sites["main_body_p0"]),
            site_pos(self._bounds_sites["main_body_px"]),
            site_pos(self._bounds_sites["main_body_py"]),
            site_pos(self._bounds_sites["main_body_pz"]),
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
            site_names = [
                f"{prefix}_p0",
                f"{prefix}_px",
                f"{prefix}_py",
                f"{prefix}_pz",
            ]
            # only return results if site names exist in model
            if site_names[0] not in self._bounds_sites:
                continue

            sites = [site_pos(self._bounds_sites[name]) for name in site_names]

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
