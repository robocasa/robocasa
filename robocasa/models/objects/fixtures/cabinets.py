import numpy as np
from copy import deepcopy
from xml.etree import ElementTree as ET

from robosuite.utils.mjcf_utils import array_to_string as a2s, find_elements, find_parent
from robosuite.utils.mjcf_utils import xml_path_completion, new_geom
from robocasa.models.objects.fixtures.fixture import ProcGenFixture, get_texture_name_from_file
from robocasa.utils.object_utils import set_geom_dimensions

from robocasa.models.objects.fixtures.cabinet_panels import *

import robocasa.utils.object_utils as OU
from robocasa.utils.object_utils import get_fixture_to_point_rel_offset

import robocasa



class Cabinet(ProcGenFixture):
    """
    Cabinet class. Procedurally defined with primitive geoms
    """
    def __init__(
        self,
        xml,
        name,
        size, # format: [w, d, h]
        thickness=0.03,
        door_gap=0.003,

        handle_type="bar",
        handle_config=None,

        panel_type="raised",  # shaker, slab, raised
        panel_config=None,

        open_top=False,  # remove top element

        texture=None,
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

        # place and size each component
        self.geoms = None
        self._create_cab()

        # remove top geom if necessary
        if self.geoms is not None and "top" in self.geoms and open_top:
            for elem in self.geoms["top"]:
                self._remove_element(elem)

        self._set_texture()

    def _set_texture(self):
        if self.texture is None:
            return
        
        self.texture = xml_path_completion(self.texture, root=robocasa.models.assets_root)
        
        texture = find_elements(
            self.root, tags="texture", 
            attribs={"name": "tex"},
            return_first=True
        )
        tex_is_2d = texture.get("type", None) == "2d"
        tex_name = get_texture_name_from_file(self.texture)
        if tex_is_2d:
            tex_name += "_2d"
        texture.set("name", tex_name)
        texture.set("file", self.texture)

        material = find_elements(
            self.root, tags="material", 
            attribs={"name": "{}_mat".format(self.name)},
            return_first=True
        )
        material.set("texture", tex_name)

    def get_reset_regions(self, env):
        p0, px, py, pz = self.get_int_sites()
        return {
            "bottom": {
                "offset": (0, 0, p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        }
    
    def _create_cab(self):
        raise NotImplementedError()

    def _add_door(
        self, w, h, th, pos, parent_body,
        handle_hpos, handle_vpos,
        door_name="door"
    ):
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
            size=[w - dg, th, h - dg], # apply door gap to width and height
            name="{}_{}".format(self.name, door_name),
            texture=self.texture,
            handle_type=self.handle_type,
            handle_config=self.handle_config,
            **panel_config,
        )
        door_elem = door.get_obj()
        door_elem.set("pos", a2s(pos))

        self.merge_assets(door)
        parent_body.append(door_elem)

    def set_door_state(self, min, max, env, rng):
        pass

    def get_door_state(self, env):
        return {}
    
    @property
    def nat_lang(self):
        return "cabinet"


class SingleCabinet(Cabinet):
    def __init__(
        self,
        name="single_cab",
        orientation="right",
        *args,
        **kwargs,
    ):
        assert orientation in ["left", "right"]
        self.orientation = orientation
        self.cabinet_type = "single"

        xml = "fixtures/cabinets/cabinet_single.xml"

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs,
        )

    def _get_cab_components(self):
        geom_names = ["top", "bottom", "back", "right", "left", "shelf", "door"]
        body_names = ["hingedoor"]
        joint_names = ["doorhinge"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self): 
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
            "back": [x - 2*th, th, z - 2*th],
            "left": [th, y - th, z - 2*th],
            "right": [th, y - th, z - 2*th],
            "shelf": [x - 2*th, y - 0.05, th],
        }
        positions = {
            "top": [0, th, z - th],
            "bottom": [0, th, -z + th],
            "back": [0, y - th, 0],
            "left": [-x + th, th, 0],
            "right": [x - th, th, 0],
            "shelf": [0, 0.05 - th, 0],
        }
        set_geom_dimensions(sizes, positions, self.geoms, rotated=True)


        # cabinet door bodies and joints
        bodies["hingedoor"].set("pos", a2s([0, 0, 0]))
        # set joint position
        if self.orientation == "left":
            joints["doorhinge"].set("pos", a2s([-x + th, -y, 0]))
            joints["doorhinge"].set("range", a2s([-3.00, 0]))
        else:
            joints["doorhinge"].set("pos", a2s([x - th, -y, 0]))

        # create door
        door_pos = [0, -y + th, 0]
        handle_hpos = "right" if self.orientation == "left" else "left"
        handle_vpos = self.panel_config.get("handle_vpos", "bottom")

        self._add_door(
            w=x * 2, h=z * 2, th=th * 2,
            pos=door_pos,
            parent_body=bodies["hingedoor"],
            handle_hpos=handle_hpos,
            handle_vpos=handle_vpos,
        )

        # set sites
        self.set_bounds_sites({
            "ext_p0": [-x, -y, -z],
            "ext_px": [x, -y , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y, z],

            "int_p0": [-x + th*2, -y + th*2, -z + th*2],
            "int_px": [x - th*2, -y + th*2, -z + th*2],
            "int_py": [-x + th*2, y - th*2, -z + th*2],
            "int_pz": [-x + th*2, -y + th*2, z - th*2],
        })

    def set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        
        joint_min = 0
        joint_max = np.pi / 2
        
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        sign = -1 if self.orientation == "left" else 1
        
        env.sim.data.set_joint_qpos(
            "{}_doorhinge".format(self.name),
            sign * rng.uniform(desired_min, desired_max)
        )

    def get_door_state(self, env):
        sim = env.sim
        hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_doorhinge")]
        sign = -1 if self.orientation == "left" else 1
        hinge_qpos = hinge_qpos * sign
        
        # convert to percentages
        door = OU.normalize_joint_value(hinge_qpos, joint_min=0, joint_max=np.pi / 2)
        
        return {
            "door": door,
        }
    
    @property
    def handle_name(self):
        return "{}_door_handle_handle".format(self.name)

    @property
    def door_name(self):
        return "{}_hingedoor".format(self.name)


class HingeCabinet(Cabinet):
    def __init__(
        self,
        name="hinge_cab",
        *args,
        **kwargs,
    ):
        self.cabinet_type = "hinge"

        xml = "fixtures/cabinets/cabinet_hinge.xml"

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs,
        )

    def _get_cab_components(self):
        """
        searches for and returns all geoms used for hinge cabinets
        """

        geom_names = [
            "top", "bottom", "back", "right", "left", "shelf",
        ]
        body_names = ["hingeleftdoor", "hingerightdoor"]
        joint_names = ["leftdoorhinge", "rightdoorhinge"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)
    
    def _create_cab(self):
        # divide sizes by two according to mujoco conventions
        x, y, z = [dim / 2 if dim is not None else None for dim in self.size]
        th = self.thickness / 2

        self.geoms, bodies, joints = self._get_cab_components()

        # set bodies positions
        bodies["hingeleftdoor"].set("pos", a2s([0, 0, 0]))
        bodies["hingerightdoor"].set("pos", a2s([0, 0, 0]))

        # set joint positions
        joints["leftdoorhinge"].set("pos", a2s([-x + th, -y, 0]))
        joints["rightdoorhinge"].set("pos", a2s([x -th, -y, 0]))

        # positions
        positions = {
            "top": [0, th, z - th],
            "bottom": [0, th, -z + th],
            "back": [0, y - th, 0],
            "left": [-x + th, th, 0],
            "right": [x - th, th, 0],
            "shelf": [0, 0.05 - th, 0],
        }
        sizes = {
            "top": [x, y - th, th],
            "bottom": [x, y - th, th],
            "back": [x - 2*th, th, z - 2*th],
            "left": [th, y - th, z - 2*th],
            "right": [th, y - th, z - 2*th],
            "shelf": [x - 2*th, y - 0.05, th],
        }
        set_geom_dimensions(sizes, positions, self.geoms, rotated=True)

        # add doors
        door_x_positions = {
            "left": -x / 2,
            "right": x / 2
        }
        handle_vpos = self.panel_config.get("handle_vpos", "bottom")

        for side in ["left", "right"]:
            self._add_door(
                w=x, h=z * 2, th=th * 2,
                pos=[door_x_positions[side], -y + th, 0],
                
                parent_body=bodies["hinge{}door".format(side)],
                handle_hpos="left" if side == "right" else "right",
                handle_vpos=handle_vpos,
                door_name=side + "_door",
            )

        # set sites
        self.set_bounds_sites({
            "ext_p0": [-x, -y, -z],
            "ext_px": [x, -y , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y, z],

            "int_p0": [-x + th*2, -y + th*2, -z + th*2],
            "int_px": [x - th*2, -y + th*2, -z + th*2],
            "int_py": [-x + th*2, y - th*2, -z + th*2],
            "int_pz": [-x + th*2, -y + th*2, z - th*2],
        })

    def get_state(self, sim):
        # angle of two door joints
        state = dict()
        for j in self._joints:
            name = "{}_{}".format(self.name, j)
            addr = sim.model.get_joint_qpos_addr(name)
            state[name] = sim.data.qpos[addr]
        return state

    def set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        
        joint_min = 0
        joint_max = np.pi / 2
        
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max
        
        env.sim.data.set_joint_qpos(
            "{}_rightdoorhinge".format(self.name),
            rng.uniform(desired_min, desired_max)
        )

        env.sim.data.set_joint_qpos(
            "{}_leftdoorhinge".format(self.name),
            -rng.uniform(desired_min, desired_max)
        )

    def get_door_state(self, env):
        sim = env.sim
        right_hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_rightdoorhinge")]
        left_hinge_qpos = -sim.data.qpos[sim.model.joint_name2id(f"{self.name}_leftdoorhinge")]
        
        # convert to percentages
        left_door = OU.normalize_joint_value(left_hinge_qpos, joint_min=0, joint_max=np.pi / 2)
        right_door = OU.normalize_joint_value(right_hinge_qpos, joint_min=0, joint_max=np.pi / 2)
        
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
    def __init__(
        self,
        name="shelves",
        num_shelves=2,
        *args,
        **kwargs,
    ):
        self.num_shelves = num_shelves
        self.shelves = list()
        super().__init__(
            xml="fixtures/cabinets/cabinet_open.xml",
            name=name,
            *args,
            **kwargs,
        )
    
    def _get_cab_components(self):
        geom_names = ["top", "bottom"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_cab(self):
        # no need to divide size here
        x, y, z = self.size
        th = self.thickness

        shelf_size = [x, y, th]
        # evenly spaced, taking thickness into account
        shelf_z_positions = np.linspace(
            start=th / 2, stop=z - th / 2,
            num=self.num_shelves, endpoint=False
        ) - z / 2

        # create and position shelves
        for i in range(self.num_shelves):
            shelf_pos = [0, 0, shelf_z_positions[i]]
            shelf = CabinetShelf(
                size=shelf_size, 
                pos=shelf_pos,
                name="{}_shelf_{}".format(self.name, i),
                texture=self.texture
            )
            self.shelves.append(shelf)

            # merge shelves
            self.merge_assets(shelf)
            shelf_elem = shelf.get_obj()
            self.get_obj().append(shelf_elem)

        self.set_bounds_sites({
            "ext_p0": [-x, -y, -z],
            "ext_px": [x, -y , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y, z],

            "int_p0": [-x + th*2, -y + th*2, -z + th*2],
            "int_px": [x - th*2, -y + th*2, -z + th*2],
            "int_py": [-x + th*2, y - th*2, -z + th*2],
            "int_pz": [-x + th*2, -y + th*2, z - th*2],
        })

    @property
    def nat_lang(self):
        return "shelves"


class Drawer(Cabinet):
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

    def _get_cab_components(self):
        geom_names = [
            "top", "bottom", "back", "right", "left",
            "inner_bottom", "inner_back", "inner_right", "inner_left",
        ]
        body_names = ["inner_box"]
        joint_names = ["slidejoint"]

        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_cab(self): 
        # divide everything by 2 according to mujoco convention       
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        self.geoms, bodies, joints = self._get_cab_components()

        """
        core cabinet housing
        """

        ix = x - 2*th - 0.001 # inner box x
        iy = y - 2*th
        iz = z - 2*th - 0.001 # inner box z

        sizes = {
            "top": [x, y - th, th],
            "bottom": [x, y - th, th],
            "back": [x - 2*th, th, z - 2*th],
            "left": [th, y - th, z - 2*th],
            "right": [th, y - th, z - 2*th],

            "inner_bottom": [ix, iy, th],
            "inner_back": [ix - 2*th, th, iz - 2*th],
            "inner_left": [th, iy, iz - 2*th],
            "inner_right": [th, iy, iz - 2*th],
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
        joints["slidejoint"].set("range", a2s([-y*2, 0]))

        # create door
        door_w, door_h, door_th = x*2, z*2, th*2 # multiply by 2 to set full size
        door_pos = [0, -y + th, 0]
        self._add_door(
            w=door_w, h=door_h, th=door_th,
            pos=door_pos,
            parent_body=bodies["inner_box"],
            handle_hpos="center",
            handle_vpos="center",
        )

        self.set_bounds_sites({
            "ext_p0": [-x, -y, -z],
            "ext_px": [x, -y , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y, z],

            "int_p0": [-ix + 2*th, -iy, -iz + 2*th],
            "int_px": [ix - 2*th, -iy, -iz + 2*th],
            "int_py": [-ix + 2*th, iy - 2*th, -iz + 2*th],
            "int_pz": [-ix + 2*th, -iy, iz],
        })

    @property
    def nat_lang(self):
        return "drawer"
    
    def update_state(self, env):
        int_sites = {}
        for site in ["int_p0", "int_px", "int_py", "int_pz" ]:
            int_sites[site] = get_fixture_to_point_rel_offset(self, np.array(env.sim.data.get_site_xpos(self.naming_prefix + site)))
        self.set_bounds_sites(int_sites)

    def set_door_state(self, min, max, env, rng):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        
        joint_min = 0
        joint_max = self.size[1] * 0.55 # dont want it to fully open up
        
        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        sign = -1 
        
        env.sim.data.set_joint_qpos(
            "{}_slidejoint".format(self.name),
            sign * rng.uniform(desired_min, desired_max)
        )

    def get_door_state(self, env):
        sim = env.sim
        hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_slidejoint")]
        sign = -1 
        hinge_qpos = hinge_qpos * sign
        
        # convert to percentages
        door = OU.normalize_joint_value(hinge_qpos, joint_min=0, joint_max=self.size[1] * 0.55)
        
        return {
            "door": door,
        }
    
    @property
    def handle_name(self):
        return "{}_door_handle_handle".format(self.name)


class PanelCabinet(Cabinet):
    def __init__(
        self,
        name="panel_cab",
        solid_body=False,
        *args,
        **kwargs,
    ):
        self.cabinet_type = "panel"

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
        x, y, z = [dim / 2 for dim in self.size]
        th = self.thickness / 2

        if self.solid_body:
            geom_name = self._name + "_body"
            size = [x, y - th, z]
            pos = [0, th, 0]
            g = new_geom(name=geom_name, type="box", size=size, pos=pos, group=0, density=10, rgba="0.5 0 0 1")
            g_vis = new_geom(name=geom_name + "_visual", type="box", size=size, pos=pos, group=1, material=self._name + "_mat", density=10, conaffinity=0, contype=0, mass=1e-8)
            self._obj.append(g)
            self._obj.append(g_vis)

        ### make a door and merge in ###
        door_w, door_h, door_th = x*2, z*2, th*2 # multiply by 2 to set full size
        door_pos = [0, -y + th, 0]
        self._add_door(
            w=door_w, h=door_h, th=door_th,
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
    def __init__(
        self,
        interior_obj,
        size=None,
        padding=None, # padding amount in [[-x, x], [-y, y], [-z, z]] directions
        name="housing_cab",
        *args,
        **kwargs,
    ):
        self.cabinet_type = "housing"

        xml = "fixtures/cabinets/cabinet_housing.xml"

        self.interior_obj = None # initially set to None for superclass initialization, set later
        
        # Parse size and padding input
        if size is None and padding is None:
            raise ValueError("Must specify size or padding for housing cabinet")
        elif size is None:
            size = [None] * 3
        elif padding is None:
            padding = [[None] * 2 for _ in range(3)]

        padding = [[None, None] if p is None else p for p in padding]

        for d in range(3):
            if size[d] is None:
                if padding[d][0] is None or padding[d][1] is None:
                    raise ValueError("If size is not specified for a dimension, both padding values must be")
                else:
                    size[d] = sum(padding[d]) + interior_obj.size[d]
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
        super().set_pos(pos)
        # we have to set the postion of the interior object as well
        if self.interior_obj is not None:
            self._place_interior_obj()

    def _place_interior_obj(self):
        """
        calculates and sets the position of the interior object
        """

        # calculate and set the position of sink
        interior_origin = np.array([
            self.pos[0] + (self.padding[0][0] - self.padding[0][1]) / 2,
            self.pos[1] + (self.padding[1][0] - self.padding[1][1]) / 2,
            self.pos[2] + (self.padding[2][0] - self.padding[2][1]) / 2,
        ])

        self.interior_obj.set_origin(interior_origin)

    def _create_cab(self):
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
            "left": [self.padding[0][0] / 2, y - self.padding[1][1] / 2, z - sum(self.padding[2]) / 2],
            "right": [self.padding[0][1] / 2, y - self.padding[1][1] / 2, z - sum(self.padding[2]) / 2],
        }

        # remove walls with size <= 0
        sizes = {geom: np.array(size) for geom, size in sizes.items() if all([d > 0 for d in size])}
        positions = {geom: np.array(pos) for geom, pos in positions.items() if geom in sizes}

        # Add geoms to xml
        for geom in sizes.keys():
            geom_name = self._name + "_" + geom
            g = new_geom(name=geom_name, type="box", size=sizes[geom], pos=positions[geom], group=0, density=10, rgba="0.5 0 0 1")
            g_vis = new_geom(name=geom_name + "_visual", type="box", size=sizes[geom], pos=positions[geom], group=1, material=self._name + "_mat", density=10, conaffinity=0, contype=0, mass=1e-8)
            self._obj.append(g)
            self._obj.append(g_vis)

        # set sites
        self.set_bounds_sites({
            "ext_p0": [-x, -y, -z],
            "ext_px": [x, -y , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y, z],

            "int_p0": [-x + self.padding[0][0], -y + self.padding[1][0], -z + self.padding[2][0]],
            "int_px": [x - self.padding[0][1], -y + self.padding[1][0], -z + self.padding[2][0]],
            "int_py": [-x + self.padding[0][0], y - self.padding[1][1], -z + self.padding[2][0]],
            "int_pz": [-x + self.padding[0][0], -y + self.padding[1][0], z - self.padding[2][1]],
        })