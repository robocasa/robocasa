"""Visualize MJCF models.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
import traceback
import cv2
import mujoco
import mujoco.viewer
import numpy as np
import robosuite
from PIL import Image
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import find_elements
from robosuite.utils.mjcf_utils import string_to_array as s2a


def edit_model_xml(xml_str):
    """
    This function edits the model xml with custom changes, including resolving relative paths,
    applying changes retroactively to existing demonstration files, and other custom scripts.
    Environment subclasses should modify this function to add environment-specific xml editing features.
    Args:
        xml_str (str): Mujoco sim demonstration XML file as string
    Returns:
        str: Edited xml file as string
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


def read_model(
    xml=None,
    filepath=None,
    hide_sites=True,
    show_bbox=False,
    show_coll_geoms=False,
):
    # either xml string is provided directly or filename containing mjcf
    assert (xml is not None) + (filepath is not None) == 1

    if filepath is not None:
        with open(filepath, "r") as file:
            xml = file.read()

    xml = edit_model_xml(xml)
    root = ET.fromstring(xml)

    # add white background
    asset = find_elements(root, tags="asset")
    skybox = ET.fromstring(
        """<texture builtin="flat" height="256" rgb1="1 1 1" rgb2="1 1 1" type="skybox" width="256"/>"""
    )
    asset.append(skybox)

    # add lighting
    worldbody = find_elements(root, tags="worldbody")
    light = ET.fromstring(
        """<light pos="2.0 -2.0 2.0" dir="0.01 0.01 -1" specular="0.3 0.3 0.3" ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" directional="true" castshadow="false"/>"""
    )
    worldbody.append(light)

    # make collision geoms (in)visible
    geoms = find_elements(root, tags="geom", return_first=False)
    for g in geoms:
        if g.get("group", None) == "0":
            if show_coll_geoms:
                g.set("rgba", "1.0 0.0 0.0 0.5")
            else:
                g.set("rgba", "1.0 0.0 0.0 0.0")

    if show_bbox:
        # reg_geoms = {}
        for geom in find_elements(root, tags="geom", return_first=False):
            name = geom.get("name", None)
            if name is None:
                continue
            if not name.startswith("reg_"):
                continue

            if name == "reg_main":
                group = 3
                geom.set("rgba", "0 1 0 1.0")
            else:
                group = 2
                geom.set("rgba", "1 1 0 0.3")

            geom.set("group", str(group))

            pos = s2a(geom.get("pos"))
            size = s2a(geom.get("size"))
            points = [
                pos + [-size[0], -size[1], -size[2]],
                pos + [-size[0], -size[1], size[2]],
                pos + [-size[0], size[1], -size[2]],
                pos + [-size[0], size[1], size[2]],
                pos + [size[0], -size[1], -size[2]],
                pos + [size[0], -size[1], size[2]],
                pos + [size[0], size[1], -size[2]],
                pos + [size[0], size[1], size[2]],
            ]

            for point in points:
                ext_bbox_site = ET.fromstring(
                    """<geom type="sphere" pos="{pos}" size="0.002" rgba="{rgba}" group="{group}" />""".format(
                        pos=a2s(point),
                        rgba="0 0 0 1",
                        group=group,
                    )
                )
                worldbody.append(ext_bbox_site)

    sites = find_elements(root, tags="site", return_first=False)
    if sites is not None:
        if hide_sites:
            # hide all sites
            for site in sites:
                site.set("rgba", "0 0 0 0")
        else:
            for site in sites:
                rgba = s2a(site.get("rgba"))
                # rgba[-1] = 1.0
                site.set("rgba", a2s(rgba))

    info = {}

    # initialize model
    xml = ET.tostring(root, encoding="unicode")
    if filepath is not None:
        os.chdir(os.path.dirname(filepath))
    t = time.time()
    model = mujoco.MjModel.from_xml_string(xml)
    sim = MjSim(model)
    info["sim_load_time"] = time.time() - t

    return sim, info


def get_model_screenshot(
    sim,
    im_width=1024,
    im_height=1024,
    cam_settings=None,
):
    t = time.time()
    render_context = MjRenderContextOffscreen(sim, device_id=-1)
    render_context.vopt.geomgroup[0] = 0
    # if cam_settings is None:
    #     cam_settings = {}
    # render_context.cam.distance = cam_settings.get("distance", 1.75)
    # render_context.cam.elevation = cam_settings.get("elevation", -30)
    sim.add_render_context(render_context)

    print(info["sim_load_time"] + (time.time() - t))

    image = sim.render(width=im_width, height=im_height)[::-1]

    return image


def render_model(
    sim,
    cam_settings=None,
):
    if cam_settings is None:
        cam_settings = {}

    mujoco.viewer.launch(sim.model._model, sim.data._data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, required=True)
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="save screenshot of model in same directory as filepath",
    )
    parser.add_argument(
        "--show_bbox",
        action="store_true",
        help="visualize exterior bounding box (based on ext_ sites)",
    )
    parser.add_argument(
        "--show_coll_geoms",
        action="store_true",
        help="whether to hide collision geoms (group 0)",
    )
    args = parser.parse_args()

    # cam_settings = {
    #     "distance": 0.3,
    #     "elevation": -30,
    # }
    cam_settings = None

    if os.path.isdir(args.mjcf):
        mjcf_path_list = []
        for root, dirs, files in os.walk(args.mjcf):
            for file in files:
                if file.endswith(".xml"):
                    mjcf_path_list.append(os.path.join(root, file))
    else:
        mjcf_path_list = [args.mjcf]

    load_time_list = []
    while True:
        for mjcf_path in mjcf_path_list:
            if len(mjcf_path_list) > 1:
                print(f"Reading: {mjcf_path}")

            sim = None
            try:
                sim, info = read_model(
                    xml=None,
                    filepath=mjcf_path,
                    hide_sites=False,
                    show_bbox=args.show_bbox,
                    show_coll_geoms=args.show_coll_geoms,
                )
            except Exception as e:
                print("Exception!")
                traceback.print_exc()

            load_time = info["sim_load_time"]
            print("sim load time:", load_time)
            load_time_list.append(load_time)

            if args.screenshot:
                image = get_model_screenshot(
                    sim=sim,
                    cam_settings=cam_settings,
                )
                im = Image.fromarray(image)
                im.save("screenshot.png")
            else:
                render_model(
                    sim=sim,
                    cam_settings=cam_settings,
                )

            del sim

        if len(mjcf_path_list) > 1:
            mean = np.mean(load_time_list)
            median = np.median(load_time_list)
            print()
            print("Mean loading time: {:.4f} s".format(mean))
            print("Median loading time: {:.4f} s".format(median))
            exit()
