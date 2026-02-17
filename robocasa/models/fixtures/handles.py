import abc
import os
from xml.etree import ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion

import robocasa
from robocasa.models.objects import MujocoXMLObjectRobocasa
from robocasa.models.fixtures.fixture import get_texture_name_from_file


class Handle(MujocoXMLObjectRobocasa):
    """
    Base class for all handles attached to cabinet/drawer panels

    Args:
        name (str): Name of the handle

        xml (str): Path to the xml file of the handle

        panel_w (float): Width of the panel to attach the handle to

        panel_h (float): Height of the panel to attach the handle to

        texture (str): Path to the texture file of the handle

        orientation (str): Orientation of the handle. Can be either "horizontal" (for drawers) or "vertical"

        length (float): Length of the handle
    """

    def __init__(
        self,
        name,
        xml,
        panel_w,
        panel_h,
        texture="textures/metals/bright_metal.png",
        orientation="vertical",
        length=None,
        duplicate_collision_geoms=True,
        *args,
        **kwargs,
    ):
        # TODO fix *args, **kwargs hack which fixes issue when setting class specific values (half_size)
        super().__init__(
            xml_path_completion(xml, root=robocasa.models.assets_root),
            name=name,
            joints=None,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        self.length = length
        self.orientation = orientation
        self.texture = xml_path_completion(texture, root=robocasa.models.assets_root)
        self.panel_w = panel_w
        self.panel_h = panel_h

        # for hinge cabinets
        self.side = None

        self._create_handle()
        self._set_texture()

    @abc.abstractmethod
    def _get_components(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_handle(self, positions, sizes):
        raise NotImplementedError

    def _get_depth_ofs(self):
        return None

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

    def _set_texture(self):
        """
        Set the texture of the handle
        """
        # set texture
        texture = find_elements(
            self.root, tags="texture", attribs={"name": "tex"}, return_first=True
        )
        # Each handle type requires different properties in the texture
        # e.g. boxed handle require cube texture, flat requires 2d texture
        # hence organize textures by type
        cls_name = type(self).__name__
        tex_name = get_texture_name_from_file(self.texture)
        tex_name = f"{cls_name}_{tex_name}"
        texture.set("file", self.texture)
        texture.set("name", tex_name)

        material = find_elements(
            self.root,
            tags="material",
            attribs={"name": "{}_mat".format(self.name)},
            return_first=True,
        )
        material.set("texture", tex_name)


class BarHandle(Handle):
    """
    Creates a bar handle

    Args:
        length (float): Length of the handle

        handle_pad (float):  A minimum difference between handle length and cabinet panel height
    """

    def __init__(self, length=0.24, handle_pad=0.04, *args, **kwargs):
        self.handle_pad = handle_pad
        super().__init__(
            xml="fixtures/handles/bar_handle.xml", length=length, *args, **kwargs
        )

    def _get_components(self):
        """
        Get the geoms of the handle
        """
        geom_names = ["handle", "handle_connector_top", "handle_connector_bottom"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        Treats the three types of cabinets separately
        """

        if self.panel_h <= 0:
            raise ValueError(f"Invalid panel height: {self.panel_h}")

        if self.length <= 0:
            raise ValueError(f"Invalid handle length: {self.length}")

        # adjust handle size if necessary
        if self.panel_h < self.length + 2 * self.handle_pad:
            self.length = self.panel_h - 2 * self.handle_pad
            if self.length <= 0:
                raise ValueError(
                    f"Panel height {self.panel_h} is too small for bar handles with padding {self.handle_pad}"
                )

        offset = self.length / 2 * 0.60  # - self.connector_pad
        # distance between main handle and door
        conn_len = 0.05

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, offset]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -offset]),
        }
        sizes = {
            "handle": [0.013, self.length / 2],
            "handle_connector_top": [0.008, conn_len / 2],
            "handle_connector_bottom": [0.008, conn_len / 2],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0, 2]] = positions[
                "handle_connector_top"
            ][[2, 0]]
            positions["handle_connector_bottom"][[0, 2]] = positions[
                "handle_connector_bottom"
            ][[2, 0]]
            eulers["handle"] = [0, 1.5708, 0]

        geoms, bodies, joints = self._get_components()
        for side in positions.keys():
            for geom in geoms[side]:
                if geom is None:
                    continue
                geom.set("pos", a2s(positions[side]))
                geom.set("size", a2s(sizes[side]))

                if eulers.get(side) is not None:
                    geom.set("euler", a2s(eulers[side]))


class BoxedHandle(Handle):
    """
    Creates a boxed handle

    Args:
        length (float): Length of the handle

        handle_pad (float):  A minimum difference between handle length and cabinet panel height
    """

    def __init__(self, length=0.24, handle_pad=0.04, *args, **kwargs):
        self.handle_pad = handle_pad
        super().__init__(
            xml="fixtures/handles/boxed_handle.xml", length=length, *args, **kwargs
        )

    def _get_components(self):
        """
        Get the geoms of the handle
        """
        geom_names = ["handle", "handle_connector_top", "handle_connector_bottom"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        Treats the three types of cabinets separately
        """

        # adjust handle size if necessary
        if self.panel_h < self.length + 2 * self.handle_pad:
            self.length = self.panel_h - 2 * self.handle_pad

        conn_len = 0.05
        connector_depth = (conn_len / 2) - 0.01
        connector_zpos = (self.length / 2) - 0.01

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, connector_zpos]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -connector_zpos]),
        }
        sizes = {
            "handle": [0.01, 0.01, self.length / 2],
            "handle_connector_top": [0.01, 0.01, connector_depth],
            "handle_connector_bottom": [0.01, 0.01, connector_depth],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0, 2]] = positions[
                "handle_connector_top"
            ][[2, 0]]
            positions["handle_connector_bottom"][[0, 2]] = positions[
                "handle_connector_bottom"
            ][[2, 0]]
            eulers["handle"] = [0, 1.5708, 0]

        geoms, bodies, joints = self._get_components()
        for side in positions.keys():
            for geom in geoms[side]:
                if geom is None:
                    continue
                geom.set("pos", a2s(positions[side]))
                geom.set("size", a2s(sizes[side]))

                if eulers.get(side) is not None:
                    geom.set("euler", a2s(eulers[side]))


class KnobHandle(Handle):
    """
    Creates a knob handle
    """

    def __init__(self, handle_pad=0.07, *args, **kwargs):
        super().__init__(
            xml="fixtures/handles/knob_handle.xml",
            # length=length,
            *args,
            **kwargs,
        )

        # z-direction padding for handle from sides of cabinet
        self.handle_pad = handle_pad

    def _get_components(self):
        """
        Get the geoms of the handle
        """
        geom_names = ["handle"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        """

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -0.017, 0]),
        }
        # radius, depth
        sizes = {
            "handle": [0.015, 0.017],
        }

        geoms, bodies, joints = self._get_components()
        for side in positions.keys():
            for geom in geoms[side]:
                if geom is None:
                    continue
                geom.set("pos", a2s(positions[side]))
                geom.set("size", a2s(sizes[side]))


class VisualMeshElongatedHandle(Handle):
    def __init__(
        self,
        xml="fixtures/handles/irregularity_handle/model.xml",
        name="irregularity_handle",
        handle_pad=0.04,
        half_size=None,
        length=0.24,
        *args,
        **kwargs,
    ):
        kwargs["duplicate_collision_geoms"] = False
        self.handle_pad = handle_pad
        if half_size is None:
            # z dim determined later in create handle
            half_size = [None, None]
        self.half_size = half_size
        super().__init__(xml=xml, name=name, length=length, *args, **kwargs)

    def _create_handle(self):
        if self.panel_h < self.length + 2 * self.handle_pad:
            self.length = self.panel_h - 2 * self.handle_pad
            if self.length <= 0:
                raise ValueError(
                    f"Panel height {self.panel_h} is too small for bar handles with padding {self.handle_pad}"
                )
        self.half_size.append(self.length / 2)
        reg_main = self._get_elements_by_name(["reg_main"], [], [])[0]["reg_main"][0]
        default_size = s2a(reg_main.get("size"))
        self.half_size = [
            hs if hs is not None else ds for hs, ds in zip(self.half_size, default_size)
        ]
        scale = np.array(self.half_size) / default_size
        self.set_scale(scale)
        if self.orientation == "horizontal":
            euler = np.array([0, 1.5708, 0])
            self._obj.set("euler", a2s(euler))

    def _get_depth_ofs(self):
        return -1 * (self.half_size[1] - 0.0025)


class VisualMeshKnobHandle(Handle):
    def __init__(
        self,
        xml="fixtures/handles/irregularity_handle/model.xml",
        name="irregularity_handle",
        radius=0.015,
        depth=0.017,
        *args,
        **kwargs,
    ):
        self.radius = radius
        self.depth = depth
        kwargs["duplicate_collision_geoms"] = False
        super().__init__(xml=xml, name=name, *args, **kwargs)

    def _create_handle(self):

        half_size = [self.radius, self.depth, self.radius]
        reg_main = self._get_elements_by_name(["reg_main"], [], [])[0]["reg_main"][0]
        default_size = s2a(reg_main.get("size"))

        scale = np.array(half_size) / default_size
        self.set_scale(scale)

    def _get_depth_ofs(self):
        return -1 * (self.depth - 0.0025)
