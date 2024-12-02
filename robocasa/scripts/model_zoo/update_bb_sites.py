import argparse
import trimesh
import os
import numpy as np
import glob
from tqdm import tqdm

from robocasa.utils.model_zoo.mjcf_gen_utils import get_bb_info
from lxml import etree
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import find_elements, array_to_string, string_to_array
from robosuite.utils.transform_utils import quat2mat, convert_quat


def update_bb_sites(model_path, show_sites=False):
    coll_path = os.path.join(model_path, "collision")
    obj_files = []
    for f_name in os.listdir(coll_path):
        obj_files.append(os.path.join(coll_path, f_name))

    # get the transformation
    xml_path = os.path.join(model_path, "model.xml")
    with open(xml_path) as f:
        xml_str = f.read()
    # tree = ET.fromstring(xml_str)
    # tree = etree.fromstring(xml_str)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # find an example mesh
    mesh_elem = find_elements(root=root, tags="mesh", return_first=True)
    refquat = string_to_array(mesh_elem.get("refquat"))
    refquat = convert_quat(refquat, to="xyzw")
    refquat[
        0:3
    ] *= (
        -1
    )  # get conjugate of quaternion (see refquat section of docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html?highlight=refquat#asset-mesh)
    sc = string_to_array(mesh_elem.get("scale"))[0]
    transform = np.zeros((4, 4))
    transform[3, 3] = 1
    transform[:3, :3] = quat2mat(refquat)

    scene = trimesh.scene.scene.Scene()
    for obj_file in obj_files:
        resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(obj_file))
        model = trimesh.load(obj_file, resolver=resolver)

        # apply transformation to model
        model.apply_transform(transform)

        scene.add_geometry(model)

    # scene.show()

    bb_info = get_bb_info(scene.bounding_box)

    worldbody = root.find("worldbody")

    sites = worldbody.find("body").findall("site")
    if show_sites:
        object_body = worldbody.find("body").find("body")
        vis_site_strings = [
            """<site rgba="1 0 0 1" size="0.005" pos="0 0 0" name="bottom_site"/>""",
            """<site rgba="0 1 0 1" size="0.005" pos="0 0 0" name="top_site"/>""",
            """<site rgba="0 0 1 1" size="0.005" pos="0 0 0" name="horizontal_radius_site"/>""",
        ]
        for s in vis_site_strings:
            object_body.append(ET.fromstring(s))

        # for debugging
        vis_sites = object_body.findall("site")
        sites = sites + vis_sites

    for site in sites:
        if site.get("name") == "horizontal_radius_site":
            pos = bb_info["pos"].copy()
            pos[0] += bb_info["size"][0] / 2
            pos[1] += bb_info["size"][1] / 2
            pos = np.array([pos[0], pos[1], pos[2]])
            site.set("pos", array_to_string(pos * sc))
        elif site.get("name") == "bottom_site":
            pos = bb_info["pos"].copy()
            pos[2] -= bb_info["size"][2] / 2
            pos = np.array([pos[0], pos[1], pos[2]])
            site.set("pos", array_to_string(pos * sc))
        elif site.get("name") == "top_site":
            pos = bb_info["pos"].copy()
            pos[2] += bb_info["size"][2] / 2
            pos = np.array([pos[0], pos[1], pos[2]])
            site.set("pos", array_to_string(pos * sc))

    with open(os.path.join(model_path, "model.xml"), "wb") as f:
        tree.write(f, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to model(s)",
    )
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.folder, "*/*"))

    for p in tqdm(paths):
        if os.path.isdir(p):
            # print("Updating:", p)
            update_bb_sites(p)
