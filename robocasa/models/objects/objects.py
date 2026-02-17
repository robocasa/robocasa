import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, find_elements


class MujocoXMLObjectRobocasa(MujocoXMLObject):
    def set_scale(self, scale, obj=None):
        """
        Scales each geom, mesh, site, and body.
        Called during initialization but can also be used externally

        Args:
            scale (float or list of floats): Scale factor (1 or 3 dims)
            obj (ET.Element) Root object to apply. Defaults to root object of model
        """
        if obj is None:
            obj = self._obj

        # set scale as 3-dim np array
        if isinstance(scale, float) or isinstance(scale, int):
            scale = np.array([scale] * 3)
        elif len(scale) == 1:
            scale = np.array([scale[0]] * 3)
        scale = np.array(scale).reshape(-1)
        assert scale.shape == (3,)
        self._scale = scale

        # scale geoms
        geom_pairs = self._get_geoms(obj)
        for _, (_, element) in enumerate(geom_pairs):
            g_pos = element.get("pos")
            g_size = element.get("size")

            if g_pos is not None:
                g_pos = string_to_array(g_pos)
                g_pos *= self._scale
                element.set("pos", array_to_string(g_pos))
            if g_size is not None:
                g_size = string_to_array(g_size)
                quat = element.get("quat")
                axisangle = element.get("axisangle")
                euler = element.get("euler")

                # get rotation matrix for geom
                if quat is not None:
                    quat = T.convert_quat(
                        string_to_array(quat)
                    )  # convert to xyzw format
                    rot_mat = T.quat2mat(quat)
                elif euler is not None:
                    euler = string_to_array(euler)
                    rot_mat = T.euler2mat(euler)
                elif axisangle is not None:
                    axisangle = string_to_array(axisangle)
                    axisangle_axayaz = axisangle[3] * axisangle[0:3]
                    rot_mat = T.quat2mat(T.axisangle2quat(axisangle_axayaz))
                else:
                    rot_mat = np.eye(3)

                if len(g_size) == 3:
                    # for boxes, spheres
                    size_world_frame = rot_mat @ g_size
                    new_size_world_frame = size_world_frame * scale
                    g_size = np.abs(np.linalg.inv(rot_mat) @ new_size_world_frame)
                elif len(g_size) == 2:
                    # cylinders
                    size_world_frame = rot_mat @ np.concatenate((g_size[0:1], g_size))
                    new_size_world_frame = size_world_frame * scale
                    g_size = np.abs(np.linalg.inv(rot_mat) @ new_size_world_frame)[1:3]
                else:
                    raise ValueError
                element.set("size", array_to_string(g_size))

        # scale meshes
        meshes = self.asset.findall("mesh")
        for elem in meshes:
            m_scale = elem.get("scale")
            if m_scale is not None:
                m_scale = string_to_array(m_scale)
            else:
                m_scale = np.ones(3)

            m_scale *= self._scale
            elem.set("scale", array_to_string(m_scale))

        # scale bodies
        body_pairs = self._get_elements(obj, "body")
        for (_, elem) in body_pairs:
            b_pos = elem.get("pos")
            if b_pos is not None:
                b_pos = string_to_array(b_pos) * self._scale
                elem.set("pos", array_to_string(b_pos))

        # scale joints
        joint_pairs = self._get_elements(obj, "joint")
        for (_, elem) in joint_pairs:
            j_pos = elem.get("pos")
            if j_pos is not None:
                j_pos = string_to_array(j_pos) * self._scale
                elem.set("pos", array_to_string(j_pos))
            if elem.get("type") == "slide":
                j_range = elem.get("range", "0 0")
                j_axis = string_to_array(elem.get("axis", "0 0 1"))
                j_range = string_to_array(j_range)
                j_index = np.where(j_axis != 0)[0]
                if len(j_index) == 0:
                    raise ValueError("Joint axis must be non-zero for slide joints")
                elif len(j_index) > 1:
                    # current implementation only supports single axis joints
                    continue
                j_index = j_index[0]
                # scale range according to axis. use the minimum scale value for objects scaled unevenly
                j_range *= min(self._scale)
                elem.set("range", array_to_string(j_range))
        # scale sites
        site_pairs = self._get_elements(self.worldbody, "site")
        for (_, elem) in site_pairs:
            s_pos = elem.get("pos")
            if s_pos is not None:
                s_pos = string_to_array(s_pos) * self._scale
                elem.set("pos", array_to_string(s_pos))

            s_size = elem.get("size")
            if s_size is not None:
                s_size_np = string_to_array(s_size)
                # handle cases where size is not 3 dimensional
                if len(s_size_np) == 3:
                    s_size_np = s_size_np * self._scale
                elif len(s_size_np) == 2:
                    scale = np.array(self._scale).reshape(-1)
                    if len(scale) == 1:
                        s_size_np *= scale
                    elif len(scale) == 3:
                        s_size_np[0] *= np.mean(scale[:2])  # width
                        s_size_np[1] *= scale[2]  # height
                    else:
                        raise ValueError
                elif len(s_size_np) == 1:
                    s_size_np *= np.mean(self._scale)
                else:
                    raise ValueError
                s_size = array_to_string(s_size_np)
                elem.set("size", s_size)

    @property
    def anchor_offset(self):
        anchor_site = self.worldbody.find(
            "./body/site[@name='{}anchor_site']".format(self.naming_prefix)
        )
        if anchor_site is None:
            return None
        site_values = string_to_array(anchor_site.get("pos"))
        return np.array(site_values)

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element
        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call
        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        return self._get_elements(root, "geom", _parent)

    def _get_elements(self, root, type, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a specific type of element
        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call
        Returns:
            list: array of (parent, child) tuples where the child element is of type
        """
        # Initialize return array
        elem_pairs = []
        # If the parent exists and this is a desired element, we add this current (parent, element) combo to the output
        if _parent is not None and root.tag == type:
            elem_pairs.append((_parent, root))
        # Loop through all children elements recursively and add to pairs
        for child in root:
            elem_pairs += self._get_elements(child, type, _parent=root)

        # Return all found pairs
        return elem_pairs


class MJCFObject(MujocoXMLObjectRobocasa):
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
        priority=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.priority = priority

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        self.mjcf_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = self.postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()

        # initialize object with new xml we wrote
        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,
        )

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)

        self._regions = dict()
        self._setup_region_dict()

    def _setup_region_dict(self):
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
            # if macros.SHOW_SITES and g_name != "reg_bbox":
            #     rgba[0:3] = np.random.uniform(0, 1, (3,))
            #     rgba[-1] = 0.3
            # else:
            #     rgba[-1] = 0.0
            rgba[-1] = 0.0
            geom.set("rgba", array_to_string(rgba))

            reg_dict = dict()
            reg_pos = string_to_array(geom.get("pos"))
            reg_halfsize = string_to_array(geom.get("size"))
            if geom.get("type") == "cylinder":
                reg_halfsize = [reg_halfsize[0], reg_halfsize[0], reg_halfsize[1]]
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

    def postprocess_model_xml(self, xml_str):
        """
        New version of postprocess model xml that only replaces robosuite file paths if necessary (otherwise
        there is an error with the "max" operation)
        """

        path = os.path.split(robosuite.__file__)[0]
        path_split = path.split("/")

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through

            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        geom_pairs = super(MJCFObject, self)._get_geoms(root=root, _parent=_parent)

        # modify geoms according to the attributes
        for i, (parent, element) in enumerate(geom_pairs):
            element.set("solref", array_to_string(self.solref))
            element.set("solimp", array_to_string(self.solimp))
            element.set("density", str(self.density))
            element.set("friction", array_to_string(self.friction))
            if self.margin is not None:
                element.set("margin", str(self.margin))

            if (self.rgba is not None) and (element.get("group") == "1"):
                element.set("rgba", array_to_string(self.rgba))

            if self.priority is not None:
                # set high priorit
                element.set("priority", str(self.priority))

        return geom_pairs

    @property
    def horizontal_radius(self):
        _horizontal_radius = string_to_array(self._regions["bbox"]["elem"].get("size"))[
            0:2
        ]
        return np.linalg.norm(_horizontal_radius)

    @property
    def bottom_offset(self):
        pos = string_to_array(self._regions["bbox"]["elem"].get("pos"))
        half_size = string_to_array(self._regions["bbox"]["elem"].get("size"))
        return np.array([pos[0], pos[1], pos[2] - half_size[2]])

    @property
    def top_offset(self):
        pos = string_to_array(self._regions["bbox"]["elem"].get("pos"))
        half_size = string_to_array(self._regions["bbox"]["elem"].get("size"))
        return np.array([pos[0], pos[1], pos[2] + half_size[2]])

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = []
        reg_bbox_geom = self._regions["bbox"]["elem"]
        center = string_to_array(reg_bbox_geom.get("pos"))
        half_size = string_to_array(reg_bbox_geom.get("size"))

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),  # p0
            center + half_size * np.array([1, -1, -1]),  # px
            center + half_size * np.array([-1, 1, -1]),  # py
            center + half_size * np.array([-1, -1, 1]),  # pz
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    @property
    def size(self):
        reg_bbox_geom = self._regions["bbox"]["elem"]
        half_size = string_to_array(reg_bbox_geom.get("size"))
        return list(half_size * 2)
