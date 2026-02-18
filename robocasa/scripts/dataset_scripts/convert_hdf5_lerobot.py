import shutil
from copy import deepcopy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
import argparse
from lerobot.datasets.utils import (
    write_info,
)
from lerobot.datasets.video_utils import (
    encode_video_frames,
)
import glob
from robocasa.utils.lerobot_utils import (
    LerobotDatasetWrapper,
    reorder_hdf5_action,
    reorder_hdf5_state,
    add_groot_specific_metadata,
    save_extra_demo_info,
    save_dataset_meta,
)
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

# default fps for robosuite
FPS = 20
VIDEO_INFO = {
    "video.fps": FPS,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}


def add_task_name(lerobot_path: Path, task_name: str, task_idx: int):
    """
    Add the task name to tasks.jsonl file in the LeRobot dataset.
    """
    tasks_path = lerobot_path / "meta" / "tasks.jsonl"
    task_name_dict = {
        "task_index": task_idx,
        "task": task_name,
    }
    with open(tasks_path, "a") as f:
        f.write(json.dumps(task_name_dict) + "\n")


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        # actions_abs=[],
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(traj_len):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len - 1)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        # traj["actions_abs"].append(action_abs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def create_env_from_hdf5(hdf5_path, args):
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )
    return env


def build_task_to_id_map(hdf5_file: h5py.File) -> dict:
    """
    Build a mapping from task descriptions to unique integer IDs.
    """
    task_to_id = {}
    demos = list(hdf5_file["data"].keys())
    for demo in demos:
        demo_data = hdf5_file["data"][demo]
        ep_meta = demo_data.attrs.get("ep_meta", None)
        if ep_meta is not None:
            lang = json.loads(ep_meta)["lang"]
            if lang not in task_to_id:
                task_to_id[lang] = len(task_to_id)
    return task_to_id


def main(args):
    raw_dataset_path = Path(args.raw_dataset_path)
    data_dir = raw_dataset_path.parent

    # Clean up any existing dataset in the output directory
    lerobot_path = data_dir / "lerobot"
    if lerobot_path.exists():
        shutil.rmtree(lerobot_path)

    raw_file = h5py.File(raw_dataset_path, "r")

    img_shape = (args.camera_height, args.camera_width, 3)

    # Create LeRobot dataset with video-backed image features + required scalars
    dataset = LerobotDatasetWrapper.create(
        repo_id=data_dir / "lerobot",
        robot_type="PandaOmron",
        fps=FPS,
        features={
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "observation.images.robot0_agentview_left": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "observation.images.robot0_agentview_right": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
            "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
            "observation.state": {"dtype": "float64", "shape": (16,)},
            "action": {"dtype": "float64", "shape": (12,)},
            "next.reward": {"dtype": "float32", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
        },
        image_writer_threads=30,
        image_writer_processes=30,
    )

    modality_path = (
        Path(__file__).parent.parent.parent
        / "models/assets/groot_dataset_assets/PandaOmron_modality.json"
    )
    with open(modality_path, "r") as f:
        modality_dict = json.load(f)

    task_to_id = build_task_to_id_map(raw_file)

    demos = list(raw_file["data"].keys())
    extras_dir = data_dir / "lerobot" / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_meta(lerobot_path, raw_file)
    env = create_env_from_hdf5(raw_dataset_path, args)

    for ep_idx, demo in enumerate(tqdm(demos)):

        # save extra information not included in the dataset but needed for other uses like env creation
        save_extra_demo_info(lerobot_path, raw_file["data"], demo, ep_idx)
        demo_data = raw_file["data"][demo]
        demo_length = len(demo_data["actions"])

        # prepare initial state to reload from
        states = demo_data["states"][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_data.attrs["model_file"]
        initial_state["ep_meta"] = demo_data.attrs.get("ep_meta", None)
        traj = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=demo_data["actions"][:],
            done_mode=1,
            add_datagen_info=False,
        )

        # reorder actions and states to match modality definition for model training
        actions = reorder_hdf5_action(demo_data["actions"][:], modality_dict)
        states = reorder_hdf5_state(traj["obs"], modality_dict)

        ep_meta = demo_data.attrs.get("ep_meta", None)
        lang = None

        lang = json.loads(ep_meta)["lang"]

        # map the task name to the length of the task_to_id
        # since the name will be added to the end tasks.jsonl later
        task_name_idx = len(task_to_id)

        for i in range(demo_length):
            dataset.add_frame(
                {
                    "observation.images.robot0_eye_in_hand": traj["obs"][
                        "robot0_eye_in_hand_image"
                    ][i],
                    "observation.images.robot0_agentview_left": traj["obs"][
                        "robot0_agentview_left_image"
                    ][i],
                    "observation.images.robot0_agentview_right": traj["obs"][
                        "robot0_agentview_right_image"
                    ][i],
                    "observation.state": states[i],
                    "action": actions[i],
                    "annotation.human.task_description": np.array(
                        [task_to_id[lang]], dtype=np.int64
                    ),
                    "next.reward": np.array([traj["rewards"][i]], dtype=np.float32),
                    "next.done": np.array([traj["dones"][i]], dtype=bool),
                    "annotation.human.task_name": np.array(
                        [task_name_idx], dtype=np.int64
                    ),
                },
                task=lang,
            )

        dataset.save_episode()
    add_groot_specific_metadata(data_dir)
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=raw_dataset_path)
    task_name = env_meta["env_name"]
    add_task_name(lerobot_path, task_name, task_name_idx)

    raw_file.close()

    # remove images dir
    images_dir = data_dir / "lerobot" / "images"
    shutil.rmtree(images_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to target structure"
    )
    parser.add_argument(
        "--raw_dataset_path",
        type=str,
        required=True,
        help="Path to the raw hdf5 dataset",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_eye_in_hand",
            "robot0_agentview_left",
            "robot0_agentview_right",
        ],
        help="List of camera names to render from the environment",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="Height of the rendered camera images",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="Width of the rendered camera images",
    )
    args = parser.parse_args()
    main(args)
