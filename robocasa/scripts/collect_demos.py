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
import time
import math
import random
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robocasa.macros as macros
from robosuite import load_controller_config
from robosuite.utils.errors import RandomizationError
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import robocasa
from robocasa.models.objects.fixtures import FixtureType

import mujoco
assert mujoco.__version__ == "3.1.1", "MuJoCo version must be 3.1.1. Please run pip install mujoco==3.1.1"

def is_empty_input_spacemouse(action):
    if np.all(action[:6] == 0) and action[6] == -1 and np.all(action[7:11] == 0):
        return True
    return False

def generate_random_pos_and_ori(base_xpos, base_ori, randomness_radius):
    noisy_pos = np.copy(base_xpos)
    noisy_ori = np.copy(base_ori)

    theta = random.uniform(0, 2 * math.pi)
    sampled_r = randomness_radius * math.sqrt(random.uniform(0, 1))
    noise_x = sampled_r * math.cos(theta)
    noise_y = sampled_r * math.sin(theta)
    noisy_pos[0] += noise_x
    noisy_pos[1] += noise_y
    noisy_ori[2] += random.uniform(0, 2 * math.pi)
    return noisy_pos, noisy_ori

def collect_human_trajectory(
    env, device, arm, env_configuration, mirror_actions,
    render=True, max_fr=None,
    print_info=True, randomness_radius=None
):
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

    if randomness_radius is not None:
        has_valid_pos = False
        # Each try takes ~8s, where env.reset() is the main cause.
        for attempt in range(10):
            env.reset()
            env._load_model()
            noisy_pos, noisy_ori = generate_random_pos_and_ori(env.planned_robot_base_xpos, env.planned_robot_base_ori, randomness_radius)
            env.robots[0].robot_model.set_base_xpos(noisy_pos)
            env.robots[0].robot_model.set_base_ori(noisy_ori)

            env._initialize_sim()
            env._reset_internal()
            env.sim.forward()
            
            # We must use robot_model and its models for contact checks. Otherwise the mount/gripper may get ignored.
            has_valid_pos = (len(env.get_contacts(env.robots[0].robot_model)) == 0)
            for model in env.robots[0].robot_model.models:
                if len(env.get_contacts(model)) != 0:
                    has_valid_pos = False
                    break
            if has_valid_pos:
                break
        if not has_valid_pos:
            raise RandomizationError("Cannot generate a valid base robot pos within the radius: {}".format(randomness_radius))
    else:
        env.reset()

    ep_meta = env.get_ep_meta()
    # print(json.dumps(ep_meta, indent=4))
    lang = ep_meta.get("lang", None)
    if print_info and lang is not None:
        print("Instruction:", lang)

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    nonzero_ac_seen = False

    # Set active robot
    active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

    if active_robot.is_mobile:
        zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
    else:
        zero_action = np.array([0, 0, 0, 0, 0, 0, -1])
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Get the newest action
        input_action, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
            mirror_actions=mirror_actions,
        )

        # If action is none, then this a reset so we should break
        if input_action is None:
            discard_traj = True
            break

        if is_empty_input_spacemouse(input_action):
            if not nonzero_ac_seen:
                if render:
                    env.render()
                continue
        else:
            nonzero_ac_seen = True

        if env.robots[0].is_mobile:
            arm_actions = input_action[:6]
            # arm_actions = np.concatenate([arm_actions, ])

            # flip some actions
            arm_actions[0], arm_actions[1] = arm_actions[1], -arm_actions[0]
            arm_actions[3], arm_actions[4] = arm_actions[4], -arm_actions[3]

            base_action = input_action[7:10]
            torso_action = input_action[10:11]

            if np.abs(torso_action[0]) < 0.50:
                torso_action[:] = 0.0

            # flip some actions
            base_action[0], base_action[1] = base_action[1], -base_action[0]

            action = np.concatenate((
                arm_actions,
                np.repeat(input_action[6:7], env.robots[0].gripper[arm].dof),
                base_action,
                torso_action,
            ))
            mode_action = input_action[-1]

            env.robots[0].enable_parts(base=True, right=True, left=True, torso=True)
            if mode_action > 0:
                action = np.concatenate((action, [1]))
            else:
                action = np.concatenate((action, [-1]))
        else:
            arm_actions = input_action
            action = env.robots[0].create_action_vector(
                {
                    arm: arm_actions[:-1], 
                    f"{arm}_gripper": arm_actions[-1:]
                }
            )

        # Run environment step
        obs, _, _, _ = env.step(action)
        if render:
            env.render()

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

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # with open("/home/soroushn/tmp/model.xml", "w") as f:
        #     f.write(env.model.get_xml())
        # exit()

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None
        
    # cleanup for end of data collection episodes
    env.close()

    return ep_directory, discard_traj


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:
        
        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))
        
        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument("--robots", nargs="+", type=str, default="PandaMobile", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument(
        "--obj_groups", type=str, nargs='+', default=None, 
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file"
    )

    parser.add_argument("--camera", type=str, default=None, help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="spacemouse", choices=["keyboard", "keyboardmobile", "spacemouse", "dummy"])
    parser.add_argument("--pos-sensitivity", type=float, default=4.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=4.0, help="How much to scale rotation user inputs")
    
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"])
    parser.add_argument("--max_fr", default=30, type=int, help="If specified, limit the frame rate")

    parser.add_argument("--layout", type=int, nargs='+', default=-1)
    parser.add_argument("--style", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]: # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print("Specifying 'obj_groups' in non-kitchen environment does not have an effect.")
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = True
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_frontview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None

        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    # Create environment
    env = suite.make(
        **config,
        has_renderer=(args.renderer != "mjviewer"),
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d-%H-%M-%S')

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError

    # make a new timestamped directory
    new_dir = os.path.join(args.directory, time_str)
    os.makedirs(new_dir)

    excluded_eps = []

    # collect demonstrations
    while True:
        print()
        ep_directory, discard_traj = collect_human_trajectory(
            env, device, args.arm, args.config, mirror_actions, render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
        )

        print("Keep traj?", not discard_traj)
        
        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split('/')[-1])
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps)
