"""
Gather individual hdf5 episodes in folder and write them to parent folder, called demo.hdf5
"""

import argparse
from copy import deepcopy
import datetime
import json
import os
import time
from glob import glob
import sys
import traceback
import shutil

import h5py
import imageio
import mujoco
import numpy as np
import robosuite
from pathlib import Path

# from robosuite import load_controller_config
from termcolor import colored

import robocasa
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5
from robosuite.controllers import load_composite_controller_config


def merge_eps(session_folder):
    all_eps_directory = os.path.join(session_folder, "episodes")

    successful_episodes = []

    env_info = None
    for ep_name in os.listdir(all_eps_directory):
        try:
            ep_dir = os.path.join(all_eps_directory, ep_name)
            with open(os.path.join(ep_dir, "ep_stats.json"), "r") as file:
                ep_stats = json.load(file)

            if ep_stats["success"]:
                successful_episodes.append(ep_name)

                if env_info is None:
                    hdf5_matches = [
                        f for f in os.listdir(ep_dir) if f.endswith(".hdf5")
                    ]
                    assert len(hdf5_matches) == 1
                    ep_hdf5_path = os.path.join(ep_dir, hdf5_matches[0])
                    f = h5py.File(ep_hdf5_path)
                    env_info = f["data"].attrs["env_info"]
                    f.close()
        except:
            continue

    # if env_info is still None, infer it (hacky)
    if env_info is None:
        print(
            colored(
                f"[WARNING]: could not recover env_info. inferring best guess",
                "magenta",
            )
        )
        controller_config = load_composite_controller_config(
            controller=None,
            robot="PandaOmron",
        )
        controller_config["composite_controller_specific_configs"] = {
            "body_part_ordering": ["right", "right_gripper", "base", "torso"]
        }
        env_name = session_folder.split("_")[-1]

        # Create argument configuration
        config = {
            "env_name": env_name,
            "robots": "PandaOmron",
            "controller_configs": controller_config,
        }

        config["layout_ids"] = -1
        config["style_ids"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
        config["translucent_robot"] = True
        config["obj_instance_split"] = "A"

        env_info = json.dumps(config)

    print(
        colored(
            f"Number of successful episodes scanned: {len(successful_episodes)}",
            "yellow",
        )
    )

    # re-process individual episodes again
    for ep_name in successful_episodes:
        ep_directory = os.path.join(all_eps_directory, ep_name)
        hdf5_path = gather_demonstrations_as_hdf5(
            all_eps_directory,
            ep_directory,
            env_info,
            successful_episodes=[ep_name],
            out_name="ep_demo.hdf5",
        )
        convert_to_robomimic_format(hdf5_path)

    # copy demo.hdf5 -> demo_orig.hdf5, if the latter doesn't currently exist
    if not os.path.exists(os.path.join(session_folder, "demo_orig.hdf5")):
        shutil.copy(
            os.path.join(session_folder, "demo.hdf5"),
            os.path.join(session_folder, "demo_orig.hdf5"),
        )

    hdf5_path = gather_demonstrations_as_hdf5(
        all_eps_directory,
        session_folder,
        env_info,
        successful_episodes=successful_episodes,
        verbose=True,
        out_name="demo.hdf5",
    )
    if hdf5_path is not None:
        convert_to_robomimic_format(hdf5_path)
        print(colored(f"Dataset saved: {hdf5_path}", "green"))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    args = parser.parse_args()

    all_session_folders = []

    for root, dirs, files in os.walk(args.directory):
        for dir in dirs:
            if dir == "episodes":
                all_session_folders.append(root)

    for session_folder in all_session_folders:
        print(colored(f"\nMerging demos for {session_folder}", "yellow"))
        try:
            merge_eps(session_folder)
        except Exception as e:
            print("Exception!")
            print(traceback.format_exc())
