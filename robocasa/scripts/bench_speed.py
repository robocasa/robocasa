"""
This script has been deprecated as it relied on tianshou which was removed to resolve dependency conflicts.
If you need environment benchmarking functionality, please:
1. Use gym.vector.SubprocVectorEnv instead
2. Or implement a custom vectorization solution
3. Or pin protobuf to an older version and reinstall tianshou
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
    raise NotImplementedError(
        "bench_speed.py has been deprecated as it relied on tianshou which was removed to resolve dependency conflicts. "
        "If you need environment benchmarking functionality, please:\n"
        "1. Use gym.vector.SubprocVectorEnv instead\n"
        "2. Or implement a custom vectorization solution\n"
        "3. Or pin protobuf to an older version and reinstall tianshou"
    )
