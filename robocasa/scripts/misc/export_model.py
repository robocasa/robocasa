import argparse
import os
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import robosuite as suite
import robosuite_model_zoo
from robosuite import load_controller_config
from robosuite_model_zoo.utils.object_play_env import ObjectPlayEnv


def export_model(args):
    controller_config = load_controller_config(default_controller="OSC_POSE")

    env_kwargs = dict(
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        controller_configs=controller_config,
    )

    if args.env == "ObjectPlayEnv":
        env = ObjectPlayEnv(
            num_objects=4,
            x_range=(-0.20, 0.20),
            y_range=(-0.20, 0.20),
            obj_mjcf_path=os.path.join(
                robosuite_model_zoo.__path__[0],
                "assets/shapenet_core/mugs/5fe74bab/model.xml",
            ),
            **env_kwargs,
        )
    else:
        env = suite.make(env_name=args.env, **env_kwargs)

    env.reset()

    model_dir = os.path.expanduser(args.path)
    xml_str = env.sim.model.get_xml()

    tree = ET.fromstring(xml_str)
    root = tree

    asset = root.find("asset")

    texture_dir = os.path.join(model_dir, "textures")
    mesh_dir = os.path.join(model_dir, "meshes")
    os.makedirs(texture_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    textures = asset.findall("texture")
    for elem in textures:
        f_path = elem.get("file")
        if f_path is None:
            continue
        f_name = f_path.split("/")[-1]
        elem.set("file", os.path.join("textures", f_name))
        shutil.copyfile(f_path, os.path.join(texture_dir, f_name))

    meshes = asset.findall("mesh")
    for elem in meshes:
        f_path = elem.get("file")
        if f_path is None:
            continue
        f_name = f_path.split("/")[-1]
        elem.set("file", f_name)
        shutil.copyfile(f_path, os.path.join(mesh_dir, f_name))

    with open(os.path.join(model_dir, "model.xml"), "w") as f:
        f.write(ET.tostring(tree, encoding="unicode"))

    state = np.array(env.sim.get_state().flatten())
    with open(os.path.join(model_dir, "state.npz"), "wb") as f:
        np.savez(f, state=state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
    )
    parser.add_argument("--env", type=str, default="Lift")

    args = parser.parse_args()

    export_model(args)
