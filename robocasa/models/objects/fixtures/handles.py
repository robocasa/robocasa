import os
import abc
import numpy as np
from xml.etree import ElementTree as ET

from robocasa.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string as a2s, find_elements
from robocasa.models.objects.fixtures.fixture import get_texture_name_from_file

import robocasa


class Handle(MujocoXMLObject):
    def __init__(
        self,
        name,
        xml,
        panel_w,
        panel_h,
        texture="textures/metals/bright_metal.png",
        orientation="vertical",
        length=None,
    ):
        super().__init__(
            xml_path_completion(xml, root=robocasa.models.assets_root),
            name=name,
            joints=None,
            duplicate_collision_geoms=True,
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
        # set texture
        texture = find_elements(
            self.root, tags="texture", 
            attribs={"name": "tex"}, 
            return_first=True
        )
        tex_name = get_texture_name_from_file(self.texture)
        texture.set("file", self.texture)
        texture.set("name", tex_name)

        material = find_elements(
            self.root, tags="material", 
            attribs={"name": "{}_mat".format(self.name)},
            return_first=True
        )
        material.set("texture", tex_name)


class BarHandle(Handle):
    def __init__(
        self,
        length=0.24, 
        # connector_pad=0.05,
        handle_pad=0.04,
        *args,
        **kwargs
    ):
        # z-direction padding for top and bottom connectors
        # self.connector_pad = connector_pad
        # assert length > connector_pad * 2
        # z-direction padding for handle from sides of cabinet
        self.handle_pad = handle_pad
        
        super().__init__(
            xml="fixtures/handles/bar_handle.xml",
            length=length,
            *args, **kwargs
        )

    def _get_components(self):
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
            # if self.length < 3 * self.connector_pad:
            #     raise ValueError("Cabinet size {:.3f} is too small for " \
            #                      "bar handles.".format(self.panel_h))    

        offset = self.length / 2 * 0.60 #- self.connector_pad
        conn_len = 0.05

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, offset]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -offset])
        }
        sizes = {
            "handle": [0.013, self.length / 2],
            "handle_connector_top": [0.008, conn_len / 2],
            "handle_connector_bottom": [0.008, conn_len / 2],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0,2]] = positions["handle_connector_top"][[2,0]]
            positions["handle_connector_bottom"][[0,2]] = positions["handle_connector_bottom"][[2,0]]
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
    def __init__(
        self,
        length=0.24, 
        handle_pad=0.04,
        *args,
        **kwargs
    ):
        self.handle_pad = handle_pad
        super().__init__(
            xml="fixtures/handles/boxed_handle.xml",
            length=length,
            *args, **kwargs
        )

    def _get_components(self):
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
        connector_zpos = (self.length/2) - 0.01

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, connector_zpos]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -connector_zpos])
        }
        sizes = {
            "handle": [0.01, 0.01, self.length / 2],
            "handle_connector_top": [0.01, 0.01, connector_depth ],
            "handle_connector_bottom": [0.01, 0.01,  connector_depth],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0,2]] = positions["handle_connector_top"][[2,0]]
            positions["handle_connector_bottom"][[0,2]] = positions["handle_connector_bottom"][[2,0]]
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
    def __init__(
        self,
        handle_pad = 0.07,
        *args,
        **kwargs
    ):
        super().__init__(
            xml="fixtures/handles/knob_handle.xml",
            # length=length,
            *args, **kwargs
        )

        # z-direction padding for handle from sides of cabinet
        self.handle_pad = handle_pad

    def _get_components(self):
        geom_names = ["handle"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):        
        # calculate the positions of the handles
        # by default set handle to bottom of cabient

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -0.017, 0]),
        }
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

