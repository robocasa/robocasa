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


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    args = parser.parse_args()

    all_eps_directory = os.path.join(args.directory, "episodes")

    successful_episodes = []

    env_info = None
    for ep_dir in os.listdir(all_eps_directory):
        try:
            with open(
                os.path.join(all_eps_directory, ep_dir, "ep_stats.json"), "r"
            ) as file:
                ep_stats = json.load(file)

            if ep_stats["success"]:
                successful_episodes.append(ep_dir)

                if env_info is None:
                    f = h5py.File(
                        os.path.join(all_eps_directory, ep_dir, "ep_demo.hdf5")
                    )
                    env_info = f["data"].attrs["env_info"]
        except:
            continue

    hdf5_path = gather_demonstrations_as_hdf5(
        all_eps_directory,
        args.directory,
        env_info,
        successful_episodes=successful_episodes,
        verbose=True,
    )
    if hdf5_path is not None:
        convert_to_robomimic_format(hdf5_path)
        print(colored(f"\nDataset saved: {hdf5_path}", "green"))
