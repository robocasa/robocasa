from copy import deepcopy
from xml.etree import ElementTree as ET
import re
import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import (
    find_elements,
    find_parent,
    new_geom,
    xml_path_completion,
)

import robocasa
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures.cabinet_panels import *
from robocasa.models.fixtures.fixture import (
    ProcGenFixture,
    get_texture_name_from_file,
)
from robocasa.utils.object_utils import (
    get_fixture_to_point_rel_offset,
    set_geom_dimensions,
)

VISUAL_MESH_PANEL_PATH_REG = {
    "CabinetDoorPanel001": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel001/model.xml",
    "CabinetDoorPanel002": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel002/model.xml",
    "CabinetDoorPanel003": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel003/model.xml",
    "CabinetDoorPanel004": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel004/model.xml",
    "CabinetDoorPanel005": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel005/model.xml",
    "CabinetDoorPanel006": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel006/model.xml",
    "CabinetDoorPanel007": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel007/model.xml",
    "CabinetDoorPanel008": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel008/model.xml",
    "CabinetDoorPanel009": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel009/model.xml",
    "CabinetDoorPanel010": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel010/model.xml",
    "CabinetDoorPanel011": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel011/model.xml",
    "CabinetDoorPanel012": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel012/model.xml",
    "CabinetDoorPanel013": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel013/model.xml",
    "CabinetDoorPanel014": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel014/model.xml",
    "CabinetDoorPanel015": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel015/model.xml",
    "CabinetDoorPanel016": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel016/model.xml",
    "CabinetDoorPanel017": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel017/model.xml",
    "CabinetDoorPanel018": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel018/model.xml",
    "CabinetDoorPanel019": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel019/model.xml",
    "CabinetDoorPanel020": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel020/model.xml",
    "CabinetDoorPanel021": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel021/model.xml",
    "CabinetDoorPanel022": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel022/model.xml",
    "CabinetDoorPanel023": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel023/model.xml",
    "CabinetDoorPanel024": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel024/model.xml",
    "CabinetDoorPanel025": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel025/model.xml",
    "CabinetDoorPanel026": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel026/model.xml",
    "CabinetDoorPanel027": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel027/model.xml",
    "CabinetDoorPanel028": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel028/model.xml",
    "CabinetDoorPanel029": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel029/model.xml",
    "CabinetDoorPanel030": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel030/model.xml",
    "CabinetDoorPanel031": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel031/model.xml",
    "CabinetDoorPanel032": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel032/model.xml",
    "CabinetDoorPanel033": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel033/model.xml",
    "CabinetDoorPanel034": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel034/model.xml",
    "CabinetDoorPanel035": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel035/model.xml",
    "CabinetDoorPanel036": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel036/model.xml",
    "CabinetDoorPanel037": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel037/model.xml",
    "CabinetDoorPanel038": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel038/model.xml",
    "CabinetDoorPanel039": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel039/model.xml",
    "CabinetDoorPanel040": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel040/model.xml",
    "CabinetDoorPanel041": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel041/model.xml",
    "CabinetDoorPanel042": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel042/model.xml",
    "CabinetDoorPanel043": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel043/model.xml",
    "CabinetDoorPanel044": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel044/model.xml",
    "CabinetDoorPanel045": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel045/model.xml",
    "CabinetDoorPanel046": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel046/model.xml",
    "CabinetDoorPanel047": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel047/model.xml",
    "CabinetDoorPanel048": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel048/model.xml",
    "CabinetDoorPanel049": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel049/model.xml",
    "CabinetDoorPanel050": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel050/model.xml",
    "CabinetDoorPanel051": "fixtures/cabinets/cabinet_panels/CabinetDoorPanel051/model.xml",
    "beige_slab": "fixtures/cabinets/cabinet_panels/beige_slab_door/model.xml",
    "red_slab": "fixtures/cabinets/cabinet_panels/red_slab_door/model.xml",
    "vertical_grain": "fixtures/cabinets/cabinet_panels/vertical_grain_door/model.xml",
    "wood_slab": "fixtures/cabinets/cabinet_panels/wood_slab_door/model.xml",
}


class Cabinet(ProcGenFixture):
    """
    Cabinet class. Procedurally defined with primitive geoms

    Args:
        xml (str): path to xml file

        name (str): name of the cabinet

        size (list): size of the cabinet [w, d, h]

        thickness (float): thickness of the cabinet walls

        door_gap (float): gap for the doors applied to height and width

        handle_type (str): type of handle attached to cabinet

        handle_config (dict): configuration for handle. contains keyword arguments for handle class

        panel_type (str): type of panel used for cabinet

        panel_config (dict): configuration for panel. contains keyword arguments for panel class

        open_top (bool): whether to remove top element

        texture (str): path to texture file
    """

    def __init__(
        self,
        xml,
        name,
        size,  # format: [w, d, h]
        thickness=0.03,
        door_gap=0.003,
        handle_type="bar",
        handle_config=None,
        panel_type="raised",  # shaker, slab, raised
        panel_config=None,
        open_top=False,  # remove top element
        texture=None,
        num_levels=0,
        is_corner_cab=None,
        *args,
        **kwargs,
    ):
        if panel_config is None:
            panel_config = dict()
        self.panel_type = panel_type
        self.panel_config = panel_config

        if handle_config is None:
            handle_config = dict()
        self.handle_type = handle_type
        self.handle_config = handle_config
        self.num_levels = num_levels

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs,
        )

        # define common variables
        self.thickness = thickness
        self.door_gap = door_gap
        self.size = np.array(size)
        self.texture = texture

        # meta data to check whether cabinet is a corner cabinet
        self.is_corner_cab = is_corner_cab

        # place and size each component
        self.geoms = None
        self._create_cab()

        # remove top geom if necessary
        if self.geoms is not None and "top" in self.geoms and open_top:
            for elem in self.geoms["top"]:
                self._remove_element(elem)

        self._set_texture()

    def get_reset_region_names(self):
        return [f"level{i}" for i in range(self.num_levels)]

    def _add_init_elements(self):
        # add procedural region elements (will later have correct values set)
        for level_i in range(self.num_levels):
            region_elem_str = """<geom name="{name}" type="box" pos="{pos}" size="{size}" group="1" conaffinity="0" contype="0" rgba="0.0 1.0 0.0 0.0"/>""".format(
                name="{}_reg_level{}".format(self.name, level_i),
                pos=a2s([0.0, 0.0, 0.0]),
                size=a2s([0.0001, 0.0001, 0.0001]),
            )
            region_elem = ET.fromstring(region_elem_str)
            self._obj.append(region_elem)

    def _set_texture(self):
        """
        Set the texture for the cabinet
        """
        if self.texture is None:
            return

        self.texture = xml_path_completion(
            self.texture, root=robocasa.models.assets_root
        )

        texture = find_elements(
            self.root, tags="texture", attribs={"name": "tex"}, return_first=True
        )
        tex_is_2d = texture.get("type", None) == "2d"
        tex_name = get_texture_name_from_file(self.texture)
        if tex_is_2d:
            tex_name += "_2d"
        texture.set("name", tex_name)
        texture.set("file", self.texture)

        material = find_elements(
            self.root,
            tags="material",
            attribs={"name": "{}_mat".format(self.name)},
            return_first=True,
        )
        material.set("texture", tex_name)

    def _create_cab(self):
        raise NotImplementedError()

    def _add_door(
        self, w, h, th, pos, parent_body, handle_hpos, handle_vpos, door_name="door"
    ):
        """
        Places a door on the cabinet

        Args:
            w (float): width of the door

            h (float): height of the door

            th (float): thickness of the door

            pos (list): position of the door

            parent_body (ET.Element): parent body element

            handle_hpos (str): horizontal position of the handle

            handle_vpos (str): vertical position of the handle

            door_name (str): name of the door

        """
        if self.panel_type == "slab" or self.panel_type is None:
            panel_class = SlabCabinetPanel
        elif self.panel_type == "shaker":
            panel_class = ShakerCabinetPanel
        elif self.panel_type == "raised":
            panel_class = RaisedCabinetPanel
        elif self.panel_type == "divided_window":
            panel_class = DividedWindowCabinetPanel
        elif self.panel_type == "full_window":
            panel_class = FullWindowedCabinetPanel
        elif self.panel_type in VISUAL_MESH_PANEL_PATH_REG:
            panel_class = VisualMeshPanel
            self.panel_config["xml"] = VISUAL_MESH_PANEL_PATH_REG[self.panel_type]
        elif self.panel_type == "no_panel":
            # Partially implemented - size/pos of body will still assume panel in front
            return
        else:
            raise NotImplementedError()
        dg = self.door_gap

        panel_config = deepcopy(self.panel_config)
        panel_config["handle_hpos"] = handle_hpos
        panel_config["handle_vpos"] = handle_vpos

        door = panel_class(
            size=[w - dg, th, h - dg],  # apply door gap to width and height
            name="{}_{}".format(self.name, door_name),
            texture=self.texture,
            handle_type=self.handle_type,
            handle_config=self.handle_config,
            **panel_config,
        )
        door_elem = door.get_obj()
        door_reg_main = door._get_components().get("reg_main")
        if door_reg_main is None:
            reg_main_pos = np.array([0, 0, 0])
        else:
            reg_main_pos = s2a(door_reg_main[0].get("pos", "0 0 0"))
        door_elem.set("pos", a2s(pos - reg_main_pos))

        self.merge_assets(door)
        parent_body.append(door_elem)

    def _add_levels(self, num_levels, upper_level_indent=0.005):
        """
        helper function to add levels and set their reset regions
        """

        # divide sizes by two according to mujoco conventions
        x, y, z = [dim / 2 if dim is not None else None for dim in self.size]
        th = self.thickness / 2

        regions = {}
        total_int_height = 2 * (z - th)
        for level_i in range(num_levels):
            level_z = (-z + th) + level_i * (total_int_height / num_levels)
            level_indent = 0 if level_i == 0 else upper_level_indent
            level_pos = np.array([0, level_indent, level_z])
            level_halfsize = np.array([x - 2 * th, y - th * 2 - level_indent, th])
            if level_i > 0:
                level = CabinetShelf(
                    pos=level_pos,
                    size=level_halfsize * 2,
                    name="{}_level{}".format(self.name, level_i),
                    texture=self.texture,
                )

                # merge level
                self.merge_assets(level)
                level_elem = level.get_obj()
                self.get_obj().append(level_elem)

            region_pos = level_pos.copy()
            region_pos[2] = (-z + th) + (level_i + 0.5) * (
                total_int_height / num_levels
            )
            region_halfsize = level_halfsize.copy()
            region_halfsize[2] = total_int_height / num_levels / 2 - th
            regions[f"level{level_i}"] = {
                "pos": region_pos,
                "halfsize": region_halfsize,
            }

        self.update_regions(regions, update_elem=True)

    def set_door_state(self, min, max, env):
        pass

    @property
    def nat_lang(self):
        return "cabinet"


class SingleCabinet(Cabinet):
    """
    Creates a SingleCabinet object which is a cabinet with a single door that either opens left or right

    Args:
        orientation (str): The direction in which the cabinet opens when facing the cabinet. "left" or "right"

        name (str): name of the cabinet
    """

    def __init__(
        self,
        size,
        name="single_cab",
        orientation="right",
        num_levels=None,
        *args,
        **kwargs,
    ):
        assert orientation in ["left", "right"]
        self.orientation = orientation
        self.cabinet_type = "single"

        if num_levels is None:
            height = size[2]
            if height <= 0.60:
                num_levels = 1
            elif height <= 0.80:
                num_levels = 2
            elif height <= 1.5:
                num_levels = 3
            elif height <= 2.0:
                num_levels = 4
            else:
                num_levels = 5
        assert num_levels >= 1

        xml = "fixtures/cabinets/cabinet_single.xml"

        super().__init__(
            name=name,
            xml=xml,
            size=size,
            num_levels=num_levels,
            *args,
            **kwargs,
        )

        if self.orientation == "right":
            joint_range = (0, 1.57)
        else:
            joint_range = (-1.57, 0)

        door_joint_name = self.door_joint_names[0]
        self._joint_infos[door_joint_name]["range"] = joint_range

    def _get_cab_components(self):
        """
        Finds and returns all geoms, bodies, and joints used for single cabinets

        Returns:
            dicts for geoms, bodies, and joints, mapping names to elements
        """
        geom_names = ["top", "bottom", "back", "right", "left", "door"]
        body_names = ["hingedoor"]
        joint_names = ["doorhinge"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self):
        """
        Creates the full cabinet. This involves setting the sizes and positions for the cabinet and the door,
        creating the door class, and determining the handle orientation. This also involves calculating the exterior
        and interior bounding boxes.
        """
        # divide everything by 2 according to mujoco convention
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        # get geoms, bodies, and joints
        # TODO: is adjusting the joint necessary?
        self.geoms, bodies, joints = self._get_cab_components()

        # cabinet housing
        sizes = {
            "top": [x, y - th, th],
            "bottom": [x, y - th, th],
            "back": [x - 2 * th, th, z - 2 * th],
            "left": [th, y - th, z - 2 * th],
            "right": [th, y - th, z - 2 * th],
        }
        positions = {
            "top": [0, th, z - th],
            "bottom": [0, th, -z + th],
            "back": [0, y - th, 0],
            "left": [-x + th, th, 0],
            "right": [x - th, th, 0],
        }
        set_geom_dimensions(sizes, positions, self.geoms, rotated=True)

        # cabinet door bodies and joints
        bodies["hingedoor"].set("pos", a2s([0, 0, 0]))
        # set joint position
        if self.orientation == "right":
            joints["doorhinge"].set("pos", a2s([x - th, -y, 0]))
            joints["doorhinge"].set("range", a2s([0, 1.57]))
        else:
            joints["doorhinge"].set("pos", a2s([-x + th, -y, 0]))
            joints["doorhinge"].set("range", a2s([-1.57, 0]))

        # create door
        door_pos = [0, -y + th, 0]
        # if the door opens right the handle must be on the left side of the door and vice versa
        handle_hpos = "right" if self.orientation == "left" else "left"
        handle_vpos = self.panel_config.get("handle_vpos", "bottom")

        self._add_door(
            w=x * 2,
            h=z * 2,
            th=th * 2,
            pos=door_pos,
            parent_body=bodies["hingedoor"],
            handle_hpos=handle_hpos,
            handle_vpos=handle_vpos,
        )

        self._add_levels(num_levels=self.num_levels, upper_level_indent=0.005)
        # main body region
        self.update_regions(
            {
                "main": {
                    "pos": [0.0, 0.0, 0.0],
                    "halfsize": [x, y, z],
                },
            },
            update_elem=True,
        )

    @property
    def handle_name(self):
        return "{}_door_handle_handle".format(self.name)

    @property
    def door_name(self):
        return "{}_hingedoor".format(self.name)


class HingeCabinet(Cabinet):
    """
    Creates a HingeCabinet object which is a cabinet with two doors that open outwards

    Args:
        name (str): name of the cabinet
    """

    def __init__(
        self,
        size,
        name="hinge_cab",
        num_levels=None,
        *args,
        **kwargs,
    ):
        self.cabinet_type = "hinge"

        xml = "fixtures/cabinets/cabinet_hinge.xml"

        if num_levels is None:
            height = size[2]
            if height <= 0.60:
                num_levels = 1
            elif height <= 0.80:
                num_levels = 2
            elif height <= 1.5:
                num_levels = 3
            elif height <= 2.0:
                num_levels = 4
            else:
                num_levels = 5
        assert num_levels >= 1

        super().__init__(
            xml=xml,
            name=name,
            size=size,
            num_levels=num_levels,
            *args,
            **kwargs,
        )

    def _get_cab_components(self):
        """
        Finds and returns all geoms, bodies, and joints used for single cabinets

        Returns:
            dicts for geoms, bodies, and joints, mapping names to elements
        """

        geom_names = [
            "top",
            "bottom",
            "back",
            "right",
            "left",
        ]
        body_names = ["hingeleftdoor", "hingerightdoor"]
        joint_names = ["leftdoorhinge", "rightdoorhinge"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self):
        """
        Creates the full cabinet. This involves setting the sizes and positions for the cabinet and the doors,
        creating the door classes. This also involves calculating the exterior
        and interior bounding boxes.
        """
        # divide sizes by two according to mujoco conventions
        x, y, z = [dim / 2 if dim is not None else None for dim in self.size]
        th = self.thickness / 2

        self.geoms, bodies, joints = self._get_cab_components()

        # set bodies positions
        bodies["hingeleftdoor"].set("pos", a2s([0, 0, 0]))
        bodies["hingerightdoor"].set("pos", a2s([0, 0, 0]))

        # set joint positions
        joints["leftdoorhinge"].set("pos", a2s([-x + th, -y, 0]))
        joints["rightdoorhinge"].set("pos", a2s([x - th, -y, 0]))

        # positions
        positions = {
            "top": [0, th, z - th],
            "bottom": [0, th, -z + th],
            "back": [0, y - th, 0],
            "left": [-x + th, th, 0],
            "right": [x - th, th, 0],
        }
        sizes = {
            "top": [x, y - th, th],
            "bottom": [x, y - th, th],
            "back": [x - 2 * th, th, z - 2 * th],
            "left": [th, y - th, z - 2 * th],
            "right": [th, y - th, z - 2 * th],
        }
        set_geom_dimensions(sizes, positions, self.geoms, rotated=True)

        # add doors
        door_x_positions = {"left": -x / 2, "right": x / 2}
        handle_vpos = self.panel_config.get("handle_vpos", "bottom")

        for side in ["left", "right"]:
            self._add_door(
                w=x,
                h=z * 2,
                th=th * 2,
                pos=[door_x_positions[side], -y + th, 0],
                parent_body=bodies["hinge{}door".format(side)],
                handle_hpos="left" if side == "right" else "right",
                handle_vpos=handle_vpos,
                door_name=side + "_door",
            )

        self._add_levels(num_levels=self.num_levels, upper_level_indent=0.005)
        # main body region
        self.update_regions(
            {
                "main": {
                    "pos": [0.0, 0.0, 0.0],
                    "halfsize": [x, y, z],
                },
            },
            update_elem=True,
        )

    def get_state(self, sim):
        """
        Args:
            env (MujocoEnv): environment

        Returns:
            dict: maps joint names to joint values
        """
        # angle of two door joints
        state = dict()
        for j in self._joints:
            name = "{}_{}".format(self.name, j)
            addr = sim.model.get_joint_qpos_addr(name)
            state[name] = sim.data.qpos[addr]
        return state

    def set_door_state(self, min, max, env):
        """
        Sets how open the doors are. Chooses a random amount between min and max.
        Min and max are percentages of how open the doors are

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (MujocoEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = np.pi / 2

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        env.sim.data.set_joint_qpos(
            "{}_rightdoorhinge".format(self.name),
            env.rng.uniform(desired_min, desired_max),
        )

        env.sim.data.set_joint_qpos(
            "{}_leftdoorhinge".format(self.name),
            -env.rng.uniform(desired_min, desired_max),
        )

    def get_door_state(self, env):
        """
        Args:
            env (MujocoEnv): environment
        Returns:
            dict: maps door name to a percentage of how open the door is (0.0 closed, 1.0 fully open)
        """
        sim = env.sim

        # Get raw qpos values
        left_qpos = sim.data.qpos[
            sim.model.get_joint_qpos_addr(f"{self.name}_leftdoorhinge")
        ]
        right_qpos = sim.data.qpos[
            sim.model.get_joint_qpos_addr(f"{self.name}_rightdoorhinge")
        ]

        # Normalize each based on correct direction
        left_door = OU.normalize_joint_value(abs(left_qpos), 0, np.pi / 2)
        right_door = OU.normalize_joint_value(abs(right_qpos), 0, np.pi / 2)

        return {
            "left_door": left_door,
            "right_door": right_door,
        }

    @property
    def left_handle_name(self):
        return "{}_left_door_handle_handle".format(self.name)

    @property
    def right_handle_name(self):
        return "{}_right_door_handle_handle".format(self.name)


class OpenCabinet(Cabinet):
    """
    Creates a OpenCabinet object which is a cabinet with open shelves

    Args:
        name (str): name of the cabinet

        num_levels (int): number of shelves in the cabinet
    """

    def __init__(
        self,
        name="shelves",
        num_levels=2,
        *args,
        **kwargs,
    ):
        super().__init__(
            xml="fixtures/cabinets/cabinet_open.xml",
            name=name,
            num_levels=num_levels,
            *args,
            **kwargs,
        )

    def _get_cab_components(self):
        """
        Finds and returns all geoms, bodies, and joints used for open cabinets

        Returns:
            dicts for geoms, bodies, and joints, mapping names to elements
        """
        geom_names = ["top", "bottom"]
        return self._get_elements_by_name(geom_names)[0]

    def get_reset_regions(self, env=None, shelf_level=None, z_range=(0.45, 1.50)):
        """
        Returns a dictionary containing reset regions for specified shelf levels within a z_range.
        Each region is defined by its world-space offset (the top surface of the shelf), its planar size, and its height.

        Args:
            env (Kitchen): the kitchen environment (unused here, but provided for API consistency).
            shelf_level (int or None): index of the shelf to use. If None, returns all shelves within z_range.
                0   → the bottom shelf (level0),
                1   → the next shelf (level1),
                …,
                -1  → the topmost shelf,
                -2  → one below the topmost, etc.
            z_range (tuple): optional Z bounds to filter usable regions by height
                (only used when shelf_level is None)

        Returns:
            dict: mapping the region's name to a dict with keys:
                "offset": (x, y, z) world-coordinate at the top surface of that shelf,
                "size":   (sx, sy) planar dimensions of the shelf top,
                "height": vertical thickness of the shelf surface.
        """

        regions = []
        for key, reg in self._regions.items():
            m = re.fullmatch(r"level(\d+)", key)
            if not m:
                continue
            idx = int(m.group(1))
            p0, px, py, pz = reg["p0"], reg["px"], reg["py"], reg["pz"]

            # Check z_range if shelf_level is not specified
            if shelf_level is None and z_range is not None:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    continue

            regions.append((idx, key, p0, px, py, pz))

        if not regions:
            raise ValueError(
                f"No 'levelX' regions found for '{self.name}' within z_range {z_range}"
            )

        regions.sort(key=lambda x: x[0])
        indices = [r[0] for r in regions]
        names = [r[1] for r in regions]

        # If shelf_level is specified, return only that shelf
        if shelf_level is not None:
            n = len(regions)
            pos = shelf_level if shelf_level >= 0 else n + shelf_level
            if pos < 0 or pos >= n:
                raise IndexError(
                    f"shelf_level {shelf_level} out of range (0..{n-1} or negative)"
                )
            idx, key, p0, px, py, pz = regions[pos]

            offset = (
                float(np.mean((p0[0], px[0]))),
                float(np.mean((p0[1], py[1]))),
                float(p0[2]),
            )
            size = (float(px[0] - p0[0]), float(py[1] - p0[1]))
            height = float(pz[2] - p0[2])

            return {key: {"offset": offset, "size": size, "height": height}}

        # Otherwise, return all regions within z_range
        reset_regions = {}
        for idx, key, p0, px, py, pz in regions:
            offset = (
                float(np.mean((p0[0], px[0]))),
                float(np.mean((p0[1], py[1]))),
                float(p0[2]),
            )
            size = (float(px[0] - p0[0]), float(py[1] - p0[1]))
            height = float(pz[2] - p0[2])

            reset_regions[key] = {"offset": offset, "size": size, "height": height}

        return reset_regions

    def _create_cab(self):
        """
        Creates the full cabinet. This involves setting the sizes and positions for each level.
        This also involves calculating the exterior and interior bounding boxes.
        """
        # no need to divide size here
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        regions = {
            "main": {
                "pos": [0.0, 0.0, 0.0],
                "halfsize": [x, y, z],
            },
        }
        total_int_height = 2 * (z - th)
        for level_i in range(self.num_levels):
            level_z = (-z + th) + level_i * (total_int_height / self.num_levels)
            level_pos = np.array([0.0, 0.0, level_z])
            level_halfsize = np.array([x, y, th])
            level = CabinetShelf(
                pos=level_pos,
                size=level_halfsize * 2,
                name="{}_level{}".format(self.name, level_i),
                texture=self.texture,
            )

            # merge level
            self.merge_assets(level)
            level_elem = level.get_obj()
            self.get_obj().append(level_elem)

            region_pos = level_pos.copy()
            region_pos[2] = (-z + th) + (level_i + 0.5) * (
                total_int_height / self.num_levels
            )
            region_halfsize = level_halfsize.copy()
            region_halfsize[2] = total_int_height / self.num_levels / 2 - th
            regions[f"level{level_i}"] = {
                "pos": region_pos,
                "halfsize": region_halfsize,
            }

        self.update_regions(regions, update_elem=True)

    @property
    def nat_lang(self):
        return "shelves"


class Drawer(Cabinet):
    """
    Creates a Drawer

    Args:
        name (str): name of the cabinet

        handle_config (dict): configuration for handle. contains keyword arguments for handle class
    """

    def __init__(
        self,
        name="drawer",
        handle_config=None,
        *args,
        **kwargs,
    ):
        self.cabinet_type = "drawer"

        xml = "fixtures/cabinets/drawer.xml"

        if handle_config is None:
            handle_config = dict()
        handle_config["orientation"] = "horizontal"

        super().__init__(
            xml=xml,
            name=name,
            handle_config=handle_config,
            *args,
            **kwargs,
        )

    def get_reset_region_names(self):
        return ("int",)

    def _get_cab_components(self):
        """
        Finds and returns all geoms, bodies, and joints used for drawers
        returns:
            dicts for geoms, bodies, and joints, mapping names to elements
        """
        geom_names = [
            "top",
            "bottom",
            "back",
            "right",
            "left",
            "inner_bottom",
            "inner_back",
            "inner_right",
            "inner_left",
        ]
        body_names = ["inner_box"]
        joint_names = ["slidejoint"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self):
        """
        Creates the full cabinet. This involves setting the sizes and positions for the cabinet and the door,
        creating the door class. This also involves calculating the exterior and interior bounding boxes.
        """
        # divide everything by 2 according to mujoco convention
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        self.geoms, bodies, joints = self._get_cab_components()

        """
        core cabinet housing
        """

        ix = x - 2 * th - 0.001  # inner box x
        iy = y - 2 * th
        iz = z - 2 * th - 0.001  # inner box z

        sizes = {
            "top": [x, y - th, th],
            "bottom": [x, y - th, th],
            "back": [x - 2 * th, th, z - 2 * th],
            "left": [th, y - th, z - 2 * th],
            "right": [th, y - th, z - 2 * th],
            "inner_bottom": [ix, iy, th],
            "inner_back": [ix - 2 * th, th, iz - 2 * th],
            "inner_left": [th, iy, iz - 2 * th],
            "inner_right": [th, iy, iz - 2 * th],
        }
        positions = {
            "top": [0, th, z - th],
            "bottom": [0, th, -z + th],
            "back": [0, y - th, 0],
            "left": [-x + th, th, 0],
            "right": [x - th, th, 0],
            "inner_bottom": [0, 0, -iz + th],
            "inner_back": [0, iy - th, 0],
            "inner_left": [-ix + th, 0, 0],
            "inner_right": [ix - th, 0, 0],
        }
        set_geom_dimensions(sizes, positions, self.geoms, rotated=True)

        # door body and joints
        bodies["inner_box"].set("pos", a2s([0, 0, 0]))
        # set joint position
        joints["slidejoint"].set("pos", a2s([0, -y, 0]))
        joints["slidejoint"].set("range", a2s([-y * 2, 0]))

        # create door
        door_w, door_h, door_th = x * 2, z * 2, th * 2  # multiply by 2 to set full size
        door_pos = [0, -y + th, 0]
        self._add_door(
            w=door_w,
            h=door_h,
            th=door_th,
            pos=door_pos,
            parent_body=bodies["inner_box"],
            handle_hpos="center",
            handle_vpos="center",
        )

        int_p0 = np.array(
            [
                -ix + 2 * th,
                -iy,
                -iz + 2 * th,
            ]
        )
        int_p1 = np.array(
            [
                ix - 2 * th,
                iy - 2 * th,
                iz,
            ]
        )

        self.update_regions(
            {
                "main": {
                    "pos": [0.0, 0.0, 0.0],
                    "halfsize": [x, y, z],
                },
                "int": {
                    "pos": (int_p0 + int_p1) / 2,
                    "halfsize": (int_p1 - int_p0) / 2,
                },
            },
            update_elem=True,
        )

    @property
    def nat_lang(self):
        return "drawer"

    def update_state(self, env):
        """
        Updates the interior bounding boxes of the drawer to be matched with
        how open the drawer is. This is needed when determining if an object
        is inside the drawer or when placing an object inside an open drawer.

        Args:
            env (MujocoEnv): environment
        """
        pos = get_fixture_to_point_rel_offset(
            self, env.sim.data.get_geom_xpos(f"{self.naming_prefix}reg_int")
        )
        # use prev half size since this wont change
        hs = s2a(self._regions["int"]["elem"].get("size"))
        # don't update the element. the position here is with respect
        # to the main drawer body, rather than the housing body.
        self.update_regions({"int": {"pos": pos, "halfsize": hs}}, update_elem=False)

    def open_door(self, env, min=0.9, max=1, partial_open=True):
        if partial_open:
            min *= 0.3
            max *= 0.3
        return super().open_door(env, min, max)

    def set_door_state(self, min, max, env):
        """
        Sets how open the drawer is. Chooses a random amount between min and max.
        Min and max are percentages of how open the drawer is.

        Args:
            min (float): minimum percentage of how open the drawer is

            max (float): maximum percentage of how open the drawer is

            env (MujocoEnv): environment

            rng (np.random.Generator): random number generator
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = self.size[1] * 0.55  # dont want it to fully open up

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        sign = -1

        env.sim.data.set_joint_qpos(
            "{}_slidejoint".format(self.name),
            sign * env.rng.uniform(desired_min, desired_max),
        )

    def get_door_state(self, env):
        """
        Args:
            env (MujocoEnv): environment

        Returns:
            dict: maps door name to a percentage of how open the door is
        """
        sim = env.sim
        hinge_qpos = sim.data.qpos[
            sim.model.get_joint_qpos_addr(f"{self.name}_slidejoint")
        ]
        sign = -1
        hinge_qpos = hinge_qpos * sign

        # convert to percentages
        door = OU.normalize_joint_value(
            hinge_qpos, joint_min=0, joint_max=self.size[1] * 0.55
        )

        return {
            "door": door,
        }

    @property
    def handle_name(self):
        return "{}_door_handle_handle".format(self.name)

    @property
    def door_joint_names(self):
        return [j_name for j_name in self._joint_infos if "slidejoint" in j_name]


class PanelCabinet(Cabinet):
    """
    Creates a PanelCabinet object which is a cabinet with a panel door but no handle.
    This is mainly used in a fixture stack where there is an unopenable cabinet/drawer

    Args:
        name (str): name of the cabinet

        solid_body (bool): whether to create a solid body for the cabinet behind the panel
    """

    def __init__(
        self,
        name="panel_cab",
        solid_body=False,
        *args,
        **kwargs,
    ):
        self.cabinet_type = "panel"

        # xml = "fixtures/cabinets/panel.xml"
        xml = "fixtures/cabinets/panel.xml"

        kwargs["handle_type"] = None
        self.solid_body = solid_body

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs,
        )

    def _get_cab_components(self):
        geom_names = []
        body_names = []
        joint_names = []

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self):
        """
        Creates the panel cabinet. This involves setting the sizes and positions for door, and
        if solid_body is True, creating a solid body for the cabinet behind the panel.
        """
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        if self.solid_body:
            geom_name = self._name + "_body"
            size = [x, y - th, z]
            pos = [0, th, 0]
            g = new_geom(
                name=geom_name,
                type="box",
                size=size,
                pos=pos,
                group=0,
                density=10,
                rgba="0.5 0 0 1",
            )
            g_vis = new_geom(
                name=geom_name + "_visual",
                type="box",
                size=size,
                pos=pos,
                group=1,
                material=self._name + "_mat",
                density=10,
                conaffinity=0,
                contype=0,
                mass=1e-8,
            )
            self._obj.append(g)
            self._obj.append(g_vis)

        ### make a door and merge in ###
        door_w, door_h, door_th = x * 2, z * 2, th * 2  # multiply by 2 to set full size
        door_pos = [0, -y + th, 0]
        self._add_door(
            w=door_w,
            h=door_h,
            th=door_th,
            pos=door_pos,
            parent_body=self.get_obj(),
            handle_hpos="center",
            handle_vpos="center",
        )

    def get_state(self, sim):
        # angle of two door joints
        state = dict()
        for j in self._joints:
            name = "{}_{}".format(self.name, j)
            addr = sim.model.get_joint_qpos_addr(name)
            state[name] = sim.data.qpos[addr]
        return state


class HousingCabinet(Cabinet):
    """
    Creates a HousingCabinet object which is a cabinet which is hollowed out to contain another object

    Args:
        interior_obj (Fixture): Fixture to be placed inside the cabinet

        size (list): Size of the cabinet in [x, y, z]

        padding (list): Thickness of the cabinet walls in [[-x, x], [-y, y], [-z, z]]. For each dimension, if size is specified,

        padding is optional and vice versa.

        name (str): Name of the cabinet
    """

    def __init__(
        self,
        interior_obj,
        size=None,
        padding=None,  # padding amount in [[-x, x], [-y, y], [-z, z]] directions
        interior_padding=None,
        name="housing_cab",
        *args,
        **kwargs,
    ):
        self.cabinet_type = "housing"

        xml = "fixtures/cabinets/cabinet_housing.xml"

        self.interior_obj = (
            None  # initially set to None for superclass initialization, set later
        )

        # Parse size and padding input
        if size is None and padding is None:
            raise ValueError("Must specify size or padding for housing cabinet")
        elif size is None:
            size = [None] * 3
        elif padding is None:
            padding = [[None] * 2 for _ in range(3)]

        padding = [[None, None] if p is None else p for p in padding]
        if interior_padding is None:
            interior_padding = [None, None, None]

        # print("name:", name)
        # print("size:", size)
        # print("padding:", padding)
        # print()

        for d in range(3):
            if size[d] is None:
                if padding[d][0] is None or padding[d][1] is None:
                    raise ValueError(
                        "If size is not specified for a dimension, both padding values must be"
                    )
                else:
                    size[d] = sum(padding[d]) + interior_obj.size[d]
                    if interior_padding[d] is not None:
                        size[d] += interior_padding[d]
            elif padding[d][0] is None and padding[d][1] is None:
                padding[d][0] = padding[d][1] = (size[d] - interior_obj.size[d]) / 2
            elif padding[d][0] is None:
                padding[d][0] = size[d] - interior_obj.size[d] - padding[d][1]
            elif padding[d][1] is None:
                padding[d][1] = size[d] - interior_obj.size[d] - padding[d][0]
            else:
                # Everything is specified, so check that sizes match exactly
                assert size[d] == sum(padding[d]) + interior_obj.size[d]

            # Round small fp errors to 0
            padding[d] = [i if abs(i) > 0.000001 else 0 for i in padding[d]]

            # Allow negative front padding for object to stick out further
            if size[d] < 0 or padding[d][1] < 0 or (d != 1 and padding[d][0] < 0):
                raise ValueError("Negative cab size or padding")

        self.padding = padding
        super().__init__(
            xml=xml,
            name=name,
            size=np.array(size),
            *args,
            **kwargs,
        )

        self.interior_obj = interior_obj
        self._place_interior_obj()

    def set_pos(self, pos):
        """
        Sets the position of the cabinet and the interior object

        Args:
            pos (list): position of the cabinet
        """
        super().set_pos(pos)
        # we have to set the postion of the interior object as well
        if self.interior_obj is not None:
            self._place_interior_obj()

    def _place_interior_obj(self):
        """
        calculates and sets the position of the interior object
        """

        # calculate and set the position of sink
        interior_origin = np.array(
            [
                self.pos[0] + (self.padding[0][0] - self.padding[0][1]) / 2,
                self.pos[1] + (self.padding[1][0] - self.padding[1][1]) / 2,
                self.pos[2] + (self.padding[2][0] - self.padding[2][1]) / 2,
            ]
        )

        self.interior_obj.set_origin(interior_origin)

    def _create_cab(self):
        """
        Creates the housing cabinet. This involves setting the sizes and positions for the sourrounding walls of the
        housing cabinet, and setting exterior and interior bounding box sites.
        """
        # divide sizes by two according to mujoco conventions
        x, y, z = [dim / 2 for dim in self.size]

        # positions of 5 walls according to padding
        positions = {
            "top": [0, -self.padding[1][1] / 2, z - self.padding[2][1] / 2],
            "bottom": [0, -self.padding[1][1] / 2, -z + self.padding[2][0] / 2],
            "back": [0, y - self.padding[1][1] / 2, 0],
            "left": [-x + self.padding[0][0] / 2, -self.padding[1][1] / 2, 0],
            "right": [x - self.padding[0][1] / 2, -self.padding[1][1] / 2, 0],
        }
        # sizes of 5 walls according to padding
        sizes = {
            "top": [x, y - self.padding[1][1] / 2, self.padding[2][1] / 2],
            "bottom": [x, y - self.padding[1][1] / 2, self.padding[2][0] / 2],
            "back": [x, self.padding[1][1] / 2, z],
            "left": [
                self.padding[0][0] / 2,
                y - self.padding[1][1] / 2,
                z - sum(self.padding[2]) / 2,
            ],
            "right": [
                self.padding[0][1] / 2,
                y - self.padding[1][1] / 2,
                z - sum(self.padding[2]) / 2,
            ],
        }

        # remove walls with size <= 0
        sizes = {
            geom: np.array(size)
            for geom, size in sizes.items()
            if all([d > 0 for d in size])
        }
        positions = {
            geom: np.array(pos) for geom, pos in positions.items() if geom in sizes
        }

        # Add geoms to xml
        for geom in sizes.keys():
            geom_name = self._name + "_" + geom
            g = new_geom(
                name=geom_name,
                type="box",
                size=sizes[geom],
                pos=positions[geom],
                group=0,
                density=10,
                rgba="0.5 0 0 1",
            )
            g_vis = new_geom(
                name=geom_name + "_visual",
                type="box",
                size=sizes[geom],
                pos=positions[geom],
                group=1,
                material=self._name + "_mat",
                density=10,
                conaffinity=0,
                contype=0,
                mass=1e-8,
            )
            self._obj.append(g)
            self._obj.append(g_vis)

        int_p0 = np.array(
            [
                -x + self.padding[0][0],
                -y + self.padding[1][0],
                -z + self.padding[2][0],
            ]
        )
        int_p1 = np.array(
            [
                x - self.padding[0][1],
                y - self.padding[1][1],
                z - self.padding[2][1],
            ]
        )

        self.update_regions(
            {
                "main": {
                    "pos": [0.0, 0.0, 0.0],
                    "halfsize": [x, y, z],
                },
                "int": {
                    "pos": (int_p0 + int_p1) / 2,
                    "halfsize": (int_p1 - int_p0) / 2,
                },
            },
            update_elem=True,
        )
