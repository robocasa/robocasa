import abc
import xml

from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion, get_elements

import robocasa
from robocasa.models.objects import MujocoXMLObject
from robocasa.models.fixtures.fixture import get_texture_name_from_file
from robocasa.models.fixtures.handles import *
from robocasa.utils.object_utils import set_geom_dimensions


VISUAL_MESH_ELONGATED_HANDLE_REG = {
    "CabinetHandle001": "fixtures/handles/CabinetHandle001/model.xml",
    "CabinetHandle002": "fixtures/handles/CabinetHandle002/model.xml",
    "CabinetHandle003": "fixtures/handles/CabinetHandle003/model.xml",
    "CabinetHandle004": "fixtures/handles/CabinetHandle004/model.xml",
    "CabinetHandle005": "fixtures/handles/CabinetHandle005/model.xml",
    "CabinetHandle006": "fixtures/handles/CabinetHandle006/model.xml",
    "CabinetHandle007": "fixtures/handles/CabinetHandle007/model.xml",
    "CabinetHandle008": "fixtures/handles/CabinetHandle008/model.xml",
    "CabinetHandle009": "fixtures/handles/CabinetHandle009/model.xml",
    "CabinetHandle010": "fixtures/handles/CabinetHandle010/model.xml",
    "CabinetHandle011": "fixtures/handles/CabinetHandle011/model.xml",
    "CabinetHandle012": "fixtures/handles/CabinetHandle012/model.xml",
    "CabinetHandle013": "fixtures/handles/CabinetHandle013/model.xml",
    "CabinetHandle014": "fixtures/handles/CabinetHandle014/model.xml",
    "CabinetHandle015": "fixtures/handles/CabinetHandle015/model.xml",
    "CabinetHandle017": "fixtures/handles/CabinetHandle017/model.xml",
    "CabinetHandle018": "fixtures/handles/CabinetHandle018/model.xml",
    "CabinetHandle019": "fixtures/handles/CabinetHandle019/model.xml",
    "CabinetHandle020": "fixtures/handles/CabinetHandle020/model.xml",
    "CabinetHandle021": "fixtures/handles/CabinetHandle021/model.xml",
    "CabinetHandle022": "fixtures/handles/CabinetHandle022/model.xml",
    "CabinetHandle023": "fixtures/handles/CabinetHandle023/model.xml",
    "CabinetHandle024": "fixtures/handles/CabinetHandle024/model.xml",
    "CabinetHandle025": "fixtures/handles/CabinetHandle025/model.xml",
    "CabinetHandle026": "fixtures/handles/CabinetHandle026/model.xml",
    "CabinetHandle027": "fixtures/handles/CabinetHandle027/model.xml",
    "CabinetHandle028": "fixtures/handles/CabinetHandle028/model.xml",
    "CabinetHandle029": "fixtures/handles/CabinetHandle029/model.xml",
    "CabinetHandle030": "fixtures/handles/CabinetHandle030/model.xml",
    "CabinetHandle031": "fixtures/handles/CabinetHandle031/model.xml",
    "CabinetHandle032": "fixtures/handles/CabinetHandle032/model.xml",
    "CabinetHandle033": "fixtures/handles/CabinetHandle033/model.xml",
    "CabinetHandle034": "fixtures/handles/CabinetHandle034/model.xml",
    "CabinetHandle037": "fixtures/handles/CabinetHandle037/model.xml",
    "CabinetHandle038": "fixtures/handles/CabinetHandle038/model.xml",
    "CabinetHandle039": "fixtures/handles/CabinetHandle039/model.xml",
    "CabinetHandle040": "fixtures/handles/CabinetHandle040/model.xml",
    "CabinetHandle041": "fixtures/handles/CabinetHandle041/model.xml",
    "CabinetHandle043": "fixtures/handles/CabinetHandle043/model.xml",
    "CabinetHandle044": "fixtures/handles/CabinetHandle044/model.xml",
    "CabinetHandle045": "fixtures/handles/CabinetHandle045/model.xml",
    "CabinetHandle046": "fixtures/handles/CabinetHandle046/model.xml",
    "CabinetHandle047": "fixtures/handles/CabinetHandle047/model.xml",
    "CabinetHandle048": "fixtures/handles/CabinetHandle048/model.xml",
    "CabinetHandle049": "fixtures/handles/CabinetHandle049/model.xml",
    "CabinetHandle050": "fixtures/handles/CabinetHandle050/model.xml",
    "flat": "fixtures/handles/flat_handle/model.xml",
    "irregularity": "fixtures/handles/irregularity_handle/model.xml",
    "rectangular": "fixtures/handles/rectangular_handle/model.xml",
    "protruding_rectangular": "fixtures/handles/protruding_rectangular_handle/model.xml",
}

VISUAL_MESH_KNOB_HANDLE_REG = {
    "CabinetHandle016": "fixtures/handles/CabinetHandle016/model.xml",
    "CabinetHandle035": "fixtures/handles/CabinetHandle035/model.xml",
    "CabinetHandle036": "fixtures/handles/CabinetHandle036/model.xml",
    "CabinetHandle042": "fixtures/handles/CabinetHandle042/model.xml",
    "CabinetHandle051": "fixtures/handles/CabinetHandle051/model.xml",
}


class CabinetPanel(MujocoXMLObject):
    """
    Base class for cabinet panels which are attached to the cabinet body.

    Args:
        xml (str): Path to the xml file for the cabinet panel.

        size (list): Size of the cabinet panel in [w, d, h] format.

        name (str): Name of the cabinet panel.

        handle_type (str): Type of handle to attach to the cabinet panel.

        handle_config (dict): Configuration for the handle.

        handle_hpos (str): Horizontal position of the handle.

        handle_vpos (str): Vertical position of the handle.

        texture (str): Path to the texture file for the cabinet panel.
    """

    def __init__(
        self,
        xml,
        size,  # format: [w, d, h]
        name,
        handle_type="bar",
        handle_config=None,
        handle_hpos=None,
        handle_vpos=None,
        texture=None,
        handle_hpercent=0.825,
        handle_vpercent=0.80,
        # needs to be false if using cabinets with visual meshes
        duplicate_collision_geoms=True,
    ):
        super().__init__(
            xml_path_completion(xml, root=robocasa.models.assets_root),
            name=name,
            joints=None,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        self.size = size
        self.texture = texture

        self.handle_type = handle_type
        self.handle_config = handle_config
        self.handle_hpos = handle_hpos
        self.handle_vpos = handle_vpos
        self.handle_hpercent = handle_hpercent
        self.handle_vpercent = handle_vpercent

        self._set_texture()
        self._create_panel()
        self._add_handle()

    @abc.abstractmethod
    def _get_components(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_panel(self):
        pass

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

        if isinstance(inp, xml.etree.ElementTree.Element):
            if inp.tag == "texture":
                return True

        return False

    def _set_texture(self):
        """
        Set the texture for the cabinet panel.
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
        material.set("name", "{}_mat".format(self.name))
        material.set("texture", tex_name)

    def _add_handle(self):
        """
        Add a handle to the cabinet panel. Creates the handle object, positions it based off the handle_hpos and handle_vpos,
        and appends it to the cabinet panel body.
        """
        if self.handle_type is None:
            return
        elif self.handle_type == "bar":
            handle_class = BarHandle
            vpad = 0.20
            hpad = 0.05
        elif self.handle_type == "knob":
            handle_class = KnobHandle
            vpad = 0.05
            hpad = 0.05
        elif self.handle_type == "boxed":
            handle_class = BoxedHandle
            vpad = 0.20
            hpad = 0.05
        elif self.handle_type in VISUAL_MESH_ELONGATED_HANDLE_REG:
            vpad = 0.20
            hpad = 0.05
            handle_class = VisualMeshElongatedHandle
            self.handle_config["xml"] = VISUAL_MESH_ELONGATED_HANDLE_REG[
                self.handle_type
            ]
        elif self.handle_type in VISUAL_MESH_KNOB_HANDLE_REG:
            vpad = 0.05
            hpad = 0.05
            handle_class = VisualMeshKnobHandle
            self.handle_config["xml"] = VISUAL_MESH_KNOB_HANDLE_REG[self.handle_type]
        else:
            raise NotImplementedError

        panel_w = self.size[0]
        panel_h = self.size[2]

        handle = handle_class(
            name="{}_handle".format(self.name),
            panel_w=panel_w,
            panel_h=panel_h,
            **self.handle_config,
        )
        handle_elem = handle.get_obj()

        handle_height = (
            handle.length
            if handle.length is not None and handle.orientation == "vertical"
            else 0.05
        )

        if self.handle_vpos == "bottom":
            handle_z = (-(panel_h / 2) + (handle_height / 2)) * self.handle_vpercent
        elif self.handle_vpos == "top":
            handle_z = ((panel_h / 2) - (handle_height / 2)) * self.handle_vpercent
        elif self.handle_vpos == "center":
            handle_z = 0.0
        else:
            raise NotImplementedError

        if self.handle_hpos == "left":
            handle_x = -(panel_w / 2) * self.handle_hpercent
        elif self.handle_hpos == "right":
            handle_x = (panel_w / 2) * self.handle_hpercent
        elif self.handle_hpos == "center":
            handle_x = 0.0
        else:
            raise NotImplementedError
        y_offset = (
            0
            if handle._get_depth_ofs() is None
            else handle._get_depth_ofs() - self.size[1] / 2
        )
        reg_main = find_elements(
            self.root,
            tags="geom",
            attribs={"name": f"{self.naming_prefix}reg_main"},
            return_first=True,
        )
        if reg_main is not None:
            # make sure to attach panel to reg main of the panel
            y_offset += s2a(reg_main.get("pos"))[1]
        handle_elem.set("pos", a2s([handle_x, y_offset, handle_z]))

        parent_body = self.get_obj()

        self.merge_assets(handle)
        parent_body.append(handle_elem)


class SlabCabinetPanel(CabinetPanel):
    """
    Initialize a slab cabinet panel, which is a simple flat panel.
    """

    def __init__(self, *args, **kwargs):
        xml = "fixtures/cabinets/cabinet_panels/slab.xml"
        super().__init__(xml=xml, *args, **kwargs)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel.
        """
        geom_names = ["door"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's geom
        """
        geoms = self._get_components()

        # divide by 2 for mujoco convention
        x, y, z = [dim / 2 for dim in self.size]

        sizes = {"door": [x, y, z]}
        positions = {"door": [0, 0, 0]}
        set_geom_dimensions(sizes, positions, geoms, rotated=True)


class ShakerCabinetPanel(CabinetPanel):
    """
    Initialize a shaker cabinet panel, which is a simple flat panel with a trim.

    Args:
        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trim (width/height).
    """

    def __init__(self, name, trim_th=0.02, trim_size=0.08, *args, **kwargs):
        self.trim_th = trim_th
        self.trim_size = trim_size

        xml = "fixtures/cabinets/cabinet_panels/shaker.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel. This includes the door and the 4 sorrounding trims.
        """
        geom_names = ["door", "trim_left", "trim_right", "trim_bottom", "trim_top"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door and trim geoms
        """
        # divide by 2 for mujoco convention
        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2
        trim_th, trim_size = self.trim_th / 2, self.trim_size / 2

        # position door and trims such that (0, 0, 0) is center
        door_th = y - trim_th
        door_y = trim_th / 2
        trim_y = -door_th / 2

        sizes = {
            "door": [x - trim_size * 2, door_th, z - trim_size * 2],
            "trim_left": [trim_size, trim_th, z],
            "trim_right": [trim_size, trim_th, z],
            "trim_top": [x - 2 * trim_size, trim_th, trim_size],
            "trim_bottom": [x - 2 * trim_size, trim_th, trim_size],
        }
        positions = {
            "door": [0, door_y, 0],
            "trim_left": [-x + trim_size, trim_y, 0],
            "trim_right": [x - trim_size, trim_y, 0],
            "trim_top": [0, trim_y, z - trim_size],
            "trim_bottom": [0, trim_y, -z + trim_size],
        }

        geoms = self._get_components()
        set_geom_dimensions(sizes, positions, geoms, rotated=True)


class RaisedCabinetPanel(CabinetPanel):
    """
    Initialize a raised cabinet panel, similar to the shaker panel, but with a raised door portion.

    Args:
        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trim (width/height).

        raised_gap (float): Gap between the raised portion and the sorrounding trims
    """

    def __init__(
        self, name, trim_th=0.02, trim_size=0.08, raised_gap=0.01, *args, **kwargs
    ):
        self.trim_th = trim_th
        self.trim_size = trim_size
        self.raised_gap = raised_gap

        xml = "fixtures/cabinets/cabinet_panels/raised.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel. This includes the door, the 4 sorrounding trims, and the raised portion.
        """
        geom_names = [
            "door",
            "door_raised",
            "trim_left",
            "trim_right",
            "trim_bottom",
            "trim_top",
        ]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door, trim, and raised portion geoms
        """

        # place the trims accordingly
        ShakerCabinetPanel._create_panel(self)

        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2

        trim_size, trim_th = self.trim_size / 2, self.trim_th / 2
        raised_gap = self.raised_gap / 2

        # place and size the raised portion accordingly,
        sizes = {
            "door_raised": [
                x - 2 * trim_size - 2 * raised_gap,
                trim_th,
                z - 2 * trim_size - 2 * raised_gap,
            ]
        }
        positions = {"door_raised": [0, -(y - trim_th) / 2, 0]}

        geoms = self._get_components()
        set_geom_dimensions(sizes, positions, geoms, rotated=True)


class DividedWindowCabinetPanel(CabinetPanel):
    """
    Initialize a divided window cabinet panel, which is a panel with a windowed door, trims around, and
    a 2 trims running down the middle of the window (horizontally and vertically aligned).

    Args:
        name (str): Name of the cabinet panel.

        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trims (width/height).
    """

    def __init__(self, name, trim_th=0.02, trim_size=0.08, *args, **kwargs):

        self.trim_th = trim_th
        self.trim_size = trim_size

        xml = "fixtures/cabinets/cabinet_panels/divided_window.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel. This includes the door, the 4 sorrounding trims,
        and the 2 trims running down the middle of the window.
        """
        geom_names = [
            "door",
            "trim_left",
            "trim_right",
            "trim_bottom",
            "trim_top",
            "horiz_trim",
            "vert_trim",
        ]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door, trims, and window trims
        """
        # divide by 2 for mujoco convention
        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2
        trim_th, trim_size = self.trim_th / 2, self.trim_size / 2

        # position door and trims such that (0, 0, 0) is center
        door_th = y - trim_th
        door_y = trim_th / 2
        trim_y = -door_th / 2

        sizes = {
            "door": [x - trim_size * 2, door_th, z - trim_size * 2],
            "trim_left": [trim_size, trim_th, z],
            "trim_right": [trim_size, trim_th, z],
            "trim_top": [x - 2 * trim_size, trim_th, trim_size],
            "trim_bottom": [x - 2 * trim_size, trim_th, trim_size],
            "vert_trim": [trim_size / 3.5, trim_th, z],
            "horiz_trim": [x - 2 * trim_size, trim_th, trim_size / 3.5],
        }
        positions = {
            "door": [0, door_y, 0],
            "trim_left": [-x + trim_size, trim_y, 0],
            "trim_right": [x - trim_size, trim_y, 0],
            "trim_top": [0, trim_y, z - trim_size],
            "trim_bottom": [0, trim_y, -z + trim_size],
            "vert_trim": [0, trim_y, 0],
            "horiz_trim": [0, trim_y, 0],
        }

        geoms = self._get_components()
        set_geom_dimensions(sizes, positions, geoms, rotated=True)


class FullWindowedCabinetPanel(CabinetPanel):
    """
    Initialize a full windowed cabinet panel, which is a panel with a windowed door and trims around.
    Same as the divided window panel but without the middle trims running through the window.

    Args:
        name (str): Name of the cabinet panel.

        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trims (width/height).

        opacity (float): Opacity of the window. Defaults to 0.5 to create a "frosted" effect.
    """

    def __init__(
        self, name, trim_th=0.02, trim_size=0.08, opacity=0.5, *args, **kwargs
    ):
        self.trim_th = trim_th
        self.trim_size = trim_size
        self.opacity = opacity

        xml = "fixtures/cabinets/cabinet_panels/full_window.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel. This includes the door and the 4 sorrounding trims.
        """
        geom_names = ["door", "trim_left", "trim_right", "trim_bottom", "trim_top"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door and trim geoms
        """
        # place the trims accordingly
        ShakerCabinetPanel._create_panel(self)
        self._set_opacity()

    def _set_opacity(self):
        """
        Set the opacity of the window.
        """
        transparent_mat = find_elements(
            self.root,
            tags="material",
            attribs={"name": f"{self.name}_transparent_material"},
            return_first=True,
        )
        transparent_mat.set("rgba", f"1 1 1 {self.opacity}")


class CabinetShelf(MujocoXMLObject):
    """
    Initialize a cabinet shelf, which is a simple flat panel but rotated 90 degrees.

    Args:
        size (list): Size of the cabinet shelf in [w, d, h] format.

        texture (str): Path to the texture file for the cabinet shelf.

        name (str): Name of the cabinet shelf.

        pos (list): Position of the cabinet shelf.
    """

    def __init__(
        self,
        size,  # format: [w, d, h]
        texture,
        name,
        pos=None,
    ):
        super().__init__(
            xml_path_completion(
                "fixtures/cabinets/cabinet_panels/shelf.xml",
                root=robocasa.models.assets_root,
            ),
            name=name,
            joints=None,
            duplicate_collision_geoms=True,
        )

        self.size = np.array(size)
        self.pos = pos if pos is not None else [0, 0, 0]
        self._create_panel()

        self.texture = texture
        self._set_texture()

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

        if isinstance(inp, xml.etree.ElementTree.Element):
            if inp.tag == "texture":
                return True

        return False

    def _set_texture(self):
        """
        Set the texture for the cabinet shelf.
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

    def _get_components(self):
        """
        Gets the geoms for the cabinet shelf.
        """
        geom_names = ["shelf"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet shelf. This involves setting the size and position of the panel's geom
        """
        geoms = self._get_components()

        sizes = {"shelf": self.size / 2}
        positions = {"shelf": self.pos}
        set_geom_dimensions(sizes, positions, geoms)


class VisualMeshPanel(CabinetPanel):
    def __init__(self, *args, **kwargs):
        kwargs["duplicate_collision_geoms"] = False
        super().__init__(*args, **kwargs)
        # give no mass to visual geoms
        self._set_physics_params(geom_group=1, density=0)
        self._set_physics_params(geom_group=0, density=10)

    def _get_components(self):
        """
        Gets the geoms for the cabinet panel.
        """
        geom_names = ["reg_main"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's geom
        """
        geoms = self._get_components()
        size = [1, 1, 1]
        for i, geom in geoms.items():
            # get size
            size = s2a(geom[0].get("size"))
            break

        # remove door_vis in geoms
        # divide by 2 for mujoco convention
        x, y, z = [dim / 2 for dim in self.size]

        scale = [x / size[0], y / size[1], z / size[2]]

        self.set_scale(scale)

    def _set_physics_params(
        self,
        geom_group,
        solimp=(0.9, 0.95, 0.001, 0.5, 2),
        solref=(0.02, 1),
        density=10,
        friction=(1, 0.005, 0.0001),
    ):
        all_geoms = get_elements(self._obj, "geom")
        for (_, geom) in all_geoms:
            if int(geom.get("group", "-1")) == geom_group:
                geom.set("solref", a2s(solref))
                geom.set("solimp", a2s(solimp))
                geom.set("density", str(density))
                geom.set("friction", a2s(friction))
