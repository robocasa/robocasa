"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
import datetime
import imageio
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np
import cv2

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv


def collect_human_trajectory(env, device, arm, env_configuration, video_path=None):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    env.reset()
    env.render()

    if video_writer is not None:
        video_img = env.sim.render(height=1024, width=1024)[::-1]
        video_writer.append_data(video_img)

    is_first = True

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    if device is not None:
        device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        if device is not None:
            # Get the newest action
            action, grasp = input2action(
                device=device,
                robot=active_robot,
                active_arm=arm,
                env_configuration=env_configuration,
                mirror_actions=True,
            )
        else:
            action = np.zeros(7)
            grasp = False

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        env.step(action)
        env.render()

        if video_writer is not None:
            video_img = env.sim.render(height=1024, width=1024)[::-1]
            video_writer.append_data(video_img)

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    if video_writer is not None:
        print("Saved to {}".format(video_path))
        video_writer.close()

    # cleanup for end of data collection episodes
    env.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join("/tmp"),
    )
    parser.add_argument(
        "--mjcf_path", type=str, required=True, help="path to object MJCF"
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )

    parser.add_argument("--scale", type=float, default=1.0, help="scale of mjcf object")

    parser.add_argument(
        "--record", action="store_true", help="record video while collecting demos"
    )

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
        "obj_mjcf_path": args.mjcf_path,
        "obj_scale": args.scale,
        # "num_objects": 5,
        # "x_range": (-0.15, 0.15),
        # "y_range": (-0.15, 0.15),
    }

    # Create environment
    env = ObjectPlayEnv(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        x_range=(0.0, 0.0),
        y_range=(0.0, 0.0),
        rotation=(0.0, 0.0),
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    # env = DataCollectionWrapper(env, tmp_directory)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        # env.viewer.add_keypress_callback("any", device.on_press)
        # env.viewer.add_keyup_callback("any", device.on_release)
        # env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dummy":
        device = None
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    counter = 0
    while True:
        video_path = None
        if args.record:
            video_path = os.path.join(new_dir, "ep_{}.mp4".format(counter))
        collect_human_trajectory(
            env, device, args.arm, args.config, video_path=video_path
        )
        counter += 1
