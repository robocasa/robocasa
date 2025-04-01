import h5py
import imageio
import numpy as np
import robosuite
import os

from robocasa.utils.dataset_registry import get_ds_path
from robocasa.scripts.dataset_scripts.playback_dataset import (
    get_env_metadata_from_dataset,
    playback_trajectory_with_env,
)
from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)


def playback_dataset_with_joint_control(ds_path, num_demos, video_path):
    print("Playback for", ds_path)

    f = h5py.File(ds_path)
    video_writer = imageio.get_writer(video_path, fps=20)

    env_meta = get_env_metadata_from_dataset(dataset_path=ds_path)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False

    env_kwargs["controller_configs"] = {
        "type": "HYBRID_MOBILE_BASE",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "kp": 150,
                "interpolation": None,
                "input_type": "absolute",  # was delta before
                "gripper": {"type": "GRIP"},
            },
            "torso": {"type": "JOINT_POSITION", "interpolation": "null", "kp": 2000},
            "base": {"type": "JOINT_VELOCITY", "interpolation": "null"},
        },
    }

    env = robosuite.make(**env_kwargs)

    ep_list = list(f["data"].keys())[:num_demos]
    for ep in ep_list:
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        joint_seq = f[f"data/{ep}/obs/robot0_joint_pos"][:]
        orig_acs = f["data/{}/actions".format(ep)][()]
        new_acs = []

        for t in range(len(joint_seq) - 1):
            action_dict = {
                "right": joint_seq[t + 1],
                "right_gripper": orig_acs[t, 6:7],
                "base": orig_acs[t, 7:10],
                "torso": orig_acs[t, 10:11],
            }
            ac = env.robots[0].create_action_vector(action_dict)
            new_acs.append(ac)
        new_acs = np.array(new_acs)

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states[: len(new_acs)],
            actions=new_acs,
            render=False,
            video_writer=video_writer,
            video_skip=5,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            first=False,
            verbose=False,
            camera_height=512,
            camera_width=768,
        )

    video_writer.close()


for task in list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS):
    ds_path, ds_meta = get_ds_path(task=task, ds_type="human_im", return_info=True)
    if ds_path is None:
        continue
    video_path = f"/tmp/playback_joint_control/{task}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    playback_dataset_with_joint_control(ds_path, num_demos=2, video_path=video_path)
