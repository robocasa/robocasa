import argparse
import trimesh
import os
import numpy as np
import glob
from tqdm import tqdm

from robocasa.utils.model_zoo.mjcf_gen_utils import get_bb_info
from robocasa.scripts.asset_scripts.prettify_xmls import prettify_xmls
from lxml import etree
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import find_elements, array_to_string, string_to_array
from robosuite.utils.transform_utils import quat2mat, convert_quat


def _classes_with_group_zero(root):
    """Return a set of default class names whose <geom> sets group="0"."""
    class_names = set()
    for default in root.findall(".//default[@class]"):
        geom_elem = default.find("geom")
        if geom_elem is not None and geom_elem.get("group") == "0":
            class_names.add(default.get("class"))
    return class_names


def get_collision_geom_paths(model_path: str):
    mjcf_path = os.path.join(model_path, "model.xml")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Gather class names whose default <geom> has group="0"
    cls_group0 = _classes_with_group_zero(root)

    # --- gather qualifying geoms -----------------------------------------
    collision_geoms = []
    for geom in root.findall(".//worldbody//geom[@type='mesh']"):
        if geom.get("group") == "0":
            collision_geoms.append(geom)
            continue
        class_name = geom.get("class")
        if class_name and class_name in cls_group0:
            collision_geoms.append(geom)
    # ---------------------------------------------------------------------
    mesh_names = {g.get("mesh") for g in collision_geoms if g.get("mesh")}

    # 2. Build a map from mesh-name -> file path in the <asset> section
    asset_block = root.find("asset")
    mesh_map = {}
    if asset_block is not None:
        for mesh in asset_block.findall("mesh"):
            name = mesh.get("name")
            file_path = mesh.get("file")
            if name and file_path:
                mesh_map[name] = file_path

    # 3. Look up each used mesh name in the asset map
    collision_files = []
    for name in mesh_names:
        if name in mesh_map:
            collision_files.append(os.path.join(model_path, mesh_map[name]))
        else:
            # If a mesh is referenced in worldbody but missing in assets,
            # you might want to log or raise an error here.
            print(f"Warning: mesh '{name}' not found in <asset> section")

    return collision_files


def remove_unwanted_sites(worldbody):
    UNWANTED_SITES = {
        "horizontal_radius_site",
        "top_site",
        "bottom_site",
        "ext_p0",
        "ext_px",
        "ext_py",
        "ext_pz",
    }
    for parent in worldbody.iter():
        for site in list(parent.findall("site")):
            if site.get("name") in UNWANTED_SITES:
                parent.remove(site)


def update_bb_geom(model_path, show_sites=False):
    obj_files = get_collision_geom_paths(model_path)
    if not obj_files:
        print(f"No collision meshes found in {model_path}. Skipping...")
        return

    # get the transformation
    xml_path = os.path.join(model_path, "model.xml")
    with open(xml_path) as f:
        xml_str = f.read()
    # tree = ET.fromstring(xml_str)
    # tree = etree.fromstring(xml_str)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ensure there's a <default class="region"> block
    main_default = root.find("default")
    if main_default is not None:
        region_default = main_default.find("default[@class='region']")
        if region_default is None:
            region_default = ET.SubElement(main_default, "default", {"class": "region"})
            # add the default geom under region default
            ET.SubElement(
                region_default,
                "geom",
                {"group": "1", "conaffinity": "0", "contype": "0", "rgba": "0 1 0 0"},
            )

    # find an example mesh
    mesh_elem = find_elements(root=root, tags="mesh", return_first=True)
    refquat = string_to_array(mesh_elem.get("refquat", "1.0 0.0 0.0 0.0"))
    refquat = convert_quat(refquat, to="xyzw")
    refquat[
        0:3
    ] *= (
        -1
    )  # get conjugate of quaternion (see refquat section of docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html?highlight=refquat#asset-mesh)
    sc = string_to_array(mesh_elem.get("scale", "1.0"))[0]
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
    object_body = find_elements(worldbody, tags="body", attribs={"name": "object"})

    center = bb_info["pos"] * sc
    half_size = (bb_info["size"] / 2.0) * sc

    reg_bbox = find_elements(
        root=object_body, tags="geom", attribs={"name": "reg_bbox"}, return_first=True
    )
    if reg_bbox is None:
        ET.SubElement(
            object_body,
            "geom",
            {
                "class": "region",
                "name": "reg_bbox",
                "type": "box",
                "pos": array_to_string(center),
                "size": array_to_string(half_size),
            },
        )
    else:
        reg_bbox.set("pos", array_to_string(center))
        reg_bbox.set("size", array_to_string(half_size))

    # Create a child-to-parent mapping
    parent_map = {child: parent for parent in root.iter() for child in parent}

    all_sites = find_elements(root, tags="site", return_first=False)
    all_sites = all_sites or []
    for site in all_sites:
        name = site.get("name")
        if name in [
            "horizontal_radius_site",
            "top_site",
            "bottom_site",
            "ext_p0",
            "ext_px",
            "ext_py",
            "ext_pz",
        ]:
            # delete the site
            parent = parent_map[site]
            parent.remove(site)

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

    # paths = glob.glob(os.path.join(args.folder)) #, '*/*'))
    paths = []
    for root, dirs, files in os.walk(args.folder):
        if "model.xml" in files:
            paths.append(root)

    for p in tqdm(paths):
        update_bb_geom(p)
        prettify_xmls(p)
