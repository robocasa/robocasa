"""
A script to run benchmarking on environments.
Runs several rollouts and logs the reset times and fps of the environment.
"""

import argparse
import os
import time

import numpy as np
from termcolor import colored
from tianshou.env import SubprocVectorEnv

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robocasa import ALL_KITCHEN_ENVIRONMENTS
import robocasa


def run_rollout(env, arm, env_configuration, num_steps=200, render=False):
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

    t_reset_start = time.time()
    obs = env.reset()
    # print(obs.keys())
    reset_time = time.time() - t_reset_start

    # ID = 2 always corresponds to agentview
    if render:
        env.render()

    # Loop until we get a reset from the input or the task completes

    if isinstance(env, SubprocVectorEnv):
        ac_dim = env.get_env_attr(key="action_spec", id=0)[0][0].shape
        ac_dim = list([len(env)]) + list(ac_dim)
    else:
        ac_dim = env.action_spec[0].shape

    t_start = time.time()

    for _ in range(num_steps):
        # Get the newest action
        action = np.random.normal(size=ac_dim)

        # Run environment step
        obs, _, _, _ = env.step(action)

        # import cv2
        # im = obs["robot0_agentview_left_image"]
        # cv2.imshow("offscreen render", im)
        # key = cv2.waitKey(1)

        if render:
            env.render()

    t_end = time.time()
    steps_per_sec = num_steps / (t_end - t_start)

    # # cleanup for end of data collection episodes
    # env.close()

    return reset_time, steps_per_sec


def log_info(message, color="yellow"):
    print(colored(message, color))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf_path", type=str, help="path to object MJCF")
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--n_objs", type=int, default=None)
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=None,
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
        default="robot0_agentview_right",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )

    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    def create_env():
        # Get controller config
        # controller_config = load_controller_config(default_controller=args.controller)
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots[0],
        )

        # Create argument configuration
        config = dict(
            controller_configs=controller_config,
            env_name=args.env,
            has_renderer=(not args.no_render),
            has_offscreen_renderer=(not args.no_render),
            use_camera_obs=(not args.no_render),
            render_camera=args.camera,
            ignore_done=True,
            reward_shaping=True,
            control_freq=20,
            camera_heights=84,
            camera_widths=84,
        )

        if args.env in ALL_KITCHEN_ENVIRONMENTS:
            config["camera_names"] = [
                "robot0_agentview_left",
                "robot0_agentview_right",
                # "robot0_eye_in_hand",
            ]
            config["layout_ids"] = 0
            config["style_ids"] = 0

            if args.env == "KitchenDemo" and args.n_objs is not None:
                config["num_objs"] = args.n_objs

            config["robots"] = args.robots or "PandaMobile"
        else:
            config["robots"] = args.robots or "Panda"

        env = suite.make(**config)
        return env

    if args.num_envs > 1:
        env_fns = [lambda env_i=i: create_env() for i in range(args.num_envs)]
        env = SubprocVectorEnv(env_fns)
    else:
        env = create_env()

    # collect demonstrations
    steps_per_sec_list = []
    reset_time_list = []
    for ep in range(10):
        reset_time, steps_per_sec = run_rollout(
            env, args.arm, args.config, render=False, num_steps=100
        )
        print("ep #{}".format(ep + 1))
        print("   {:.2f}s reset time".format(reset_time))
        print("   {:.2f} fps".format(steps_per_sec))
        print()
        reset_time_list.append(reset_time)
        steps_per_sec_list.append(steps_per_sec)

    print("reset time: {:.2f}s".format(np.mean(reset_time_list)))
    print("fps: {:.2f}".format(np.mean(steps_per_sec_list)))

    env.close()
