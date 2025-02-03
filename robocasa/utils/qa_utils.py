from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.scripts.playback_dataset import (
    playback_trajectory_with_env,
    get_env_metadata_from_dataset,
    reset_to,
)

import robosuite
import robosuite.utils.sim_utils as SimUtils

import argparse
import h5py
import numpy as np
from termcolor import colored
import os
import imageio
from tqdm import tqdm
import json


def scan_datset_quality(ds_path, num_demos=None):
    """
    scan for bad instances where the robot drops objects,
    collides with the environment,
    or experiences arm lock
    """
    env_meta = get_env_metadata_from_dataset(dataset_path=ds_path)
    env = create_env_from_env_meta(env_meta)

    f = h5py.File(ds_path)
    qa_stats = dict()

    demo_keys = list(f["data"].keys())
    if num_demos is not None:
        demo_keys = demo_keys[:num_demos]

    for ep in tqdm(demo_keys):
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        reset_to(env, initial_state)

        traj_len = states.shape[0]

        num_arm_lock_steps_15 = 0
        num_arm_lock_steps_5 = 0

        qa_stats[ep] = dict(
            arm_lock=[],
            obj_drop=[],
            arm_coll=[],
        )

        for t in range(traj_len):
            reset_to(env, {"states": states[t]})

            if check_obj_drop(env) is False:
                qa_stats[ep]["obj_drop"] += [t]

            if check_arm_collision(env) is False:
                qa_stats[ep]["arm_coll"] += [t]

            joint_val = read_arm_joint(env)
            if joint_val > -15:
                num_arm_lock_steps_15 += 1
            if joint_val > -5:
                num_arm_lock_steps_5 += 1

            if num_arm_lock_steps_15 > 20 or num_arm_lock_steps_5 > 5:
                qa_stats[ep]["arm_lock"] += [t]

    env.close()
    f.close()

    return qa_stats


def scan_arm_lock(ds_path):
    """
    scan for instances of arm lock.
    relies on the "robot0_joint_pos" obs key stored in the dataset to quickly test for this
    """
    f = h5py.File(ds_path)
    bad_demo_keys = []
    all_demo_keys = f["data"].keys()
    for demo_key in all_demo_keys:
        joint_vals = f[f"data/{demo_key}/obs/robot0_joint_pos"][:, 3] * 180 / np.pi
        num_arm_lock_steps_15 = np.sum((joint_vals > -15))
        num_arm_lock_steps_5 = np.sum((joint_vals > -5))

        # as long as there's less than half a second of large joint limit violation,
        # or a second of mild-to-large joint limit, violation,
        # we are ok
        if num_arm_lock_steps_15 > 20 or num_arm_lock_steps_5 > 5:
            bad_demo_keys.append(demo_key)

    f.close()

    return bad_demo_keys


def read_arm_joint(env):
    robot = env.robots[0]
    joint_vals_rad = np.array(
        [robot.sim.data.qpos[x] for x in robot._ref_joint_pos_indexes]
    )
    joint_vals_deg = joint_vals_rad * 180 / np.pi
    return joint_vals_deg[3]


def check_obj_drop(env):
    for cfg in env.object_cfgs:
        obj_name = cfg["name"]
        obj_vel = np.array(env.sim.data.get_body_xvelp(obj_name + "_main"))
        if np.any(np.abs(obj_vel) > 1.5):
            return False

    return True


def check_arm_collision(env):
    contacts = SimUtils.get_contacts(sim=env.sim, model=env.robots[0].robot_model)
    return len(contacts) == 0


def create_env_from_env_meta(env_meta):
    # create env from env meta data
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)
    return env


def playback_demos(ds_path, demo_keys, highlight_timesteps_list=None, video_path=None):
    if len(demo_keys) == 0:
        return

    f = h5py.File(ds_path)

    video_writer = imageio.get_writer(video_path, fps=20)

    for ep_i, ep in enumerate(demo_keys):
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        env_meta = get_env_metadata_from_dataset(dataset_path=ds_path)
        env = create_env_from_env_meta(env_meta)

        highlight_timesteps = None
        if highlight_timesteps_list is not None:
            highlight_timesteps = highlight_timesteps_list[ep_i]

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=None,
            render=False,
            video_writer=video_writer,
            video_skip=5,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            first=False,
            highlight_timesteps=highlight_timesteps,
            verbose=False,
        )

    f.close()
    print(colored(f"Saved video to {video_path}", "green"))
    video_writer.close()

    if env is not None:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
        nargs="+",
    )
    parser.add_argument(
        "--num_demos_to_scan",
        type=int,
        help="number of demos to scan",
        default=None,
    )
    parser.add_argument(
        "--num_demos_to_render",
        type=int,
        help="number of offending demos to visualize per test",
        default=10,
    )
    args = parser.parse_args()

    ds_path_list = args.dataset
    if args.dataset is None:
        ds_path_list = []
        for task in list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS):
            ds_path, ds_meta = get_ds_path(task=task, ds_type="mg_im", return_info=True)
            if ds_path is not None:
                ds_path_list.append(ds_path)

    for ds_path in ds_path_list:
        ds_path = os.path.expanduser(ds_path)

        # scan dataset quality
        all_stats = scan_datset_quality(ds_path, num_demos=args.num_demos_to_scan)

        summary_stats = dict()
        for ep in all_stats:
            for metric in all_stats[ep]:
                if metric not in summary_stats:
                    summary_stats[metric] = []

                flagged_timesteps = all_stats[ep][metric]
                # if at least one timestep is flagged, add this episode to list of flagged episodes for this metric
                if len(flagged_timesteps) > 0:
                    summary_stats[metric].append(ep)

        qa_path = os.path.join(os.path.dirname(ds_path), "qa")
        os.makedirs(qa_path, exist_ok=True)

        all_flagged_eps = set(
            [ep for ep_list in summary_stats.values() for ep in ep_list]
        )

        print("Path:", ds_path)
        print(
            "Total number of bad demos:",
            len(all_flagged_eps),
        )
        for metric in summary_stats:
            print(f"Num flagged demos due to {metric}: {len(summary_stats[metric])}")

        print("\nPlaying back flagged demos...")
        for metric in summary_stats:
            video_path = os.path.join(qa_path, f"{metric}.mp4")
            demo_keys = summary_stats[metric][: args.num_demos_to_render]
            highlight_timesteps_list = [all_stats[ep][metric] for ep in demo_keys]
            playback_demos(
                ds_path,
                demo_keys,
                highlight_timesteps_list=highlight_timesteps_list,
                video_path=video_path,
            )

        with open(os.path.join(qa_path, "qa_stats.json"), "w") as f:
            f.write(json.dumps(summary_stats))
