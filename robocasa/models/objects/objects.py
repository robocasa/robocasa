import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


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
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = string_to_array(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = []

        bottom_offset = self.bottom_offset
        top_offset = self.top_offset
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        horiz_radius = string_to_array(horizontal_radius_site.get("pos"))[:2]

        center = np.mean([bottom_offset, top_offset], axis=0)
        half_size = [horiz_radius[0], horiz_radius[1], top_offset[2] - center[2]]

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
