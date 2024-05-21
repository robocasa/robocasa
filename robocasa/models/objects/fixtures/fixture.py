import numpy as np
from copy import deepcopy
import abc
import random
import os
import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import (
    array_to_string,
    string_to_array,
    xml_path_completion,
    find_elements,
)
from robocasa.models.objects import MujocoXMLObject
from robocasa.utils.object_utils import get_pos_after_rel_offset
from robosuite.utils.mjcf_utils import find_parent
import robosuite.utils.transform_utils as T
import robocasa.macros as macros

import robocasa

from enum import IntEnum

def site_pos(site):
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

class Fixture(MujocoXMLObject):
    def __init__(
        self,
        xml,
        name,
        duplicate_collision_geoms=True,
        pos=None,
        scale=1,
        size=None,
        placement=None,
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
        for postfix in ["ext_p0", "ext_px", "ext_py", "ext_pz", "int_p0", "int_px", "int_py", "int_pz"]:
            site = find_elements(
                self.worldbody,
                tags="site",
                attribs={"name": "{}{}".format(self.naming_prefix, postfix)},
                return_first=True,
            )
            if site is None:
                continue
            rgba = string_to_array(site.get("rgba"))
            if macros.SHOW_SITES:
                rgba[-1] = 1.0
            else:
                rgba[-1] = 0.0
            site.set("rgba", array_to_string(rgba))
            self._bounds_sites[postfix] = site

        # scale based on specified max dimension
        if size is not None:
            self.set_scale_from_size(size)

        # based on exterior points, overwritten by subclasses (e.g. Counter) that do not have such sites
        self.size = np.array([self.width, self.depth, self.height])

        # set offset between center of object and center of exterior bounding boxes
        if self.width is not None:
            try:
                # calculate based on bounding points
                p0 = site_pos(self._bounds_sites["ext_p0"])
                px = site_pos(self._bounds_sites["ext_px"])
                py = site_pos(self._bounds_sites["ext_py"])
                pz = site_pos(self._bounds_sites["ext_pz"])
                self.origin_offset = np.array([
                    np.mean((p0[0], px[0])),
                    np.mean((p0[1], py[1])),
                    np.mean((p0[2], pz[2])),
                ])
            except KeyError:
                self.origin_offset = [0, 0, 0]
        else:
            self.origin_offset = [0, 0, 0]
        self.origin_offset = np.array(self.origin_offset)

        # placement config, for determining where to place fixture (most fixture will not use this)
        self._placement = placement

    def set_origin(self, origin):        
        # compute new position
        fixture_rot = np.array([0, 0, self.rot])
        fixture_mat = T.euler2mat(fixture_rot)
    
        pos = origin + np.dot(fixture_mat, -self.origin_offset)
        self.set_pos(pos)

    def set_scale_from_size(self, size):
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

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, each region defined as position, x_bounds, y_bounds
        """
        p0, px, py, pz = self.get_int_sites()
        return {
            "bottom": {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        }

    def sample_reset_region(self, *args, **kwargs):
        regions = self.get_reset_regions(*args, **kwargs)
        return random.sample(list(regions.values()), 1)[0]

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
        return site_pos(self._bounds_sites["ext_p0"])
    
    @property
    def width(self):
        """
        for getting the width of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_px" in self._bounds_sites:
            ext_p0 = site_pos(self._bounds_sites["ext_p0"])
            ext_px = site_pos(self._bounds_sites["ext_px"])
            w = ext_px[0] - ext_p0[0]
            return w
        else:
            return None
        
    @property
    def depth(self):
        """
        for getting the depth of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_py" in self._bounds_sites:
            ext_p0 = site_pos(self._bounds_sites["ext_p0"])
            ext_py = site_pos(self._bounds_sites["ext_py"])
            d = ext_py[1] - ext_p0[1]
            return d
        else:
            return None
        
    @property
    def height(self):
        """
        for getting the height of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_pz" in self._bounds_sites:
            ext_p0 = site_pos(self._bounds_sites["ext_p0"])
            ext_pz = site_pos(self._bounds_sites["ext_pz"])
            h = ext_pz[2] - ext_p0[2]
            return h
        else:
            return None

    def set_bounds_sites(self, pos_dict):
        for (name, pos) in pos_dict.items():
            self._bounds_sites[name].set("pos", array_to_string(pos))

    def get_ext_sites(self, all_points=False, relative=True):
        sites = [
            site_pos(self._bounds_sites["ext_p0"]),
            site_pos(self._bounds_sites["ext_px"]),
            site_pos(self._bounds_sites["ext_py"]),
            site_pos(self._bounds_sites["ext_pz"]),
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
        sites = [
            site_pos(self._bounds_sites["int_p0"]),
            site_pos(self._bounds_sites["int_px"]),
            site_pos(self._bounds_sites["int_py"]),
            site_pos(self._bounds_sites["int_pz"]),
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
