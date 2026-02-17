from lerobot.datasets.utils import (
    write_info,
)
from lerobot.datasets.video_utils import (
    encode_video_frames,
)
import glob
import shutil
from robocasa.utils.robomimic.robomimic_dataset_utils import (
    get_env_metadata_from_dataset,
)
import xml.etree.ElementTree as ET

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from pathlib import Path
import json
import gzip
import pandas as pd
from tqdm import tqdm


class LerobotDatasetWrapper(LeRobotDataset):
    """
    Wrapper class for creating LeRobotDataset. Class is needed so that we can override
    methods to get more control over video encoding parameters.
    """

    def encode_episode_videos(self, episode_index: int) -> None:
        """
        Encode videos for a given episode index. Code is mostly copied from parent class
        but with modified video new encoding parameters.
        """
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir,
                video_path,
                self.fps,
                overwrite=True,
                vcodec="h264",
                pix_fmt="yuv420p",
                crf=23,
                g=None,
                fast_decode=0,
            )
            shutil.rmtree(img_dir)

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(
                self.meta.info, self.meta.root
            )  # ensure video info always written properly


# Define the ordering for lerobot datasets
ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}

# Mapping from hdf5 state keys to lerobot dataset keys
LEROBOT_STATE_TO_HDF5_STATE = {
    "base_position": "robot0_base_pos",
    "base_rotation": "robot0_base_quat",
    "end_effector_position_relative": "robot0_base_to_eef_pos",
    "end_effector_rotation_relative": "robot0_base_to_eef_quat",
    "gripper_qpos": "robot0_gripper_qpos",
}


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """
    Calculate the dataset statistics of all columns for a list of parquet files.
    Code taken from gr00t/data/dataset.py from https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release.
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        print(f"Computing statistics for {le_modality}...")
        # check if the data is the modality is actually a list of numbers
        # skip if it is a string
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            print(f"Skipping {le_modality} because it is a string")
            continue

        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def reorder_hdf5_action(action_hdf5: np.ndarray, modality_dict) -> np.ndarray:
    """
    Reorder the action array from HDF5 dataset to LeRobot dataset format based on the modality dictionary.

    Args:
        action_hdf5 (np.ndarray): The action array in HDF5 format.
        modality_dict (dict): The modality dictionary containing action key information.

    Returns:
        np.ndarray: The reordered action array.
    """
    action_info = modality_dict["action"]
    # sort keys based on start index in action info
    sorted_action_keys = sorted(
        action_info.keys(), key=lambda k: action_info[k]["start"]
    )
    reordered_action = np.zeros_like(action_hdf5)
    for key in sorted_action_keys:
        hdf5_start, hdf5_end = ACTION_KEY_ORDERING_HDF5[key]
        lerobot_start = action_info[key]["start"]
        lerobot_end = action_info[key]["end"]
        reordered_action[:, lerobot_start:lerobot_end] = action_hdf5[
            :, hdf5_start:hdf5_end
        ]
    return reordered_action


def reorder_hdf5_state(state_hdf5: dict[str, np.ndarray], modality_dict) -> np.ndarray:
    """
    Reorder the state array from HDF5 dataset to LeRobot dataset format based on the modality dictionary.
    """
    state_info = modality_dict["state"]
    # sort keys based on start index in state info
    sorted_state_keys = sorted(state_info.keys(), key=lambda k: state_info[k]["start"])
    ep_len = len(state_hdf5["robot0_base_pos"])
    state_size = state_info[sorted_state_keys[-1]]["end"]
    reordered_state = np.zeros((ep_len, state_size), dtype=np.float64)
    for key in sorted_state_keys:
        hdf5_key = LEROBOT_STATE_TO_HDF5_STATE[key]
        lerobot_start = state_info[key]["start"]
        lerobot_end = state_info[key]["end"]
        reordered_state[:, lerobot_start:lerobot_end] = state_hdf5[hdf5_key][:]
    return reordered_state


def add_groot_specific_metadata(data_dir: str):
    """
    Adds metadata files to dataset to allow for compatibility with gr00t.
    """

    # Create lerobot directory if it doesn't exist
    lerobot_meta_dir = Path(data_dir) / "lerobot" / "meta"

    src_modality_path = (
        Path(__file__).parent.parent
        / "models/assets/groot_dataset_assets/PandaOmron_modality.json"
    )
    dst_modality_path = lerobot_meta_dir / "modality.json"
    shutil.copyfile(src_modality_path, dst_modality_path)

    src_embodiment_path = (
        Path(__file__).parent.parent
        / "models/assets/groot_dataset_assets/PandaOmron_embodiment.json"
    )
    dst_embodiment_path = lerobot_meta_dir / "embodiment.json"
    shutil.copyfile(src_embodiment_path, dst_embodiment_path)

    parquet_paths = glob.glob(
        str(Path(data_dir) / "lerobot" / "data" / "*" / "episode_*.parquet")
    )
    stats = calculate_dataset_statistics(parquet_paths)
    stats_path = lerobot_meta_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


def save_extra_demo_info(dataset_dir: Path, hdf5_dataset, demo, demo_index):
    """
    Save extra information for a demonstration episode from the HDF5 dataset. This data
    is not included in the main dataset but is needed for demonstration playback.
    """
    extras_dir = dataset_dir / "extras"
    states = hdf5_dataset[demo]["states"][:]
    ep_meta = json.loads(hdf5_dataset[demo].attrs["ep_meta"])
    model_file = hdf5_dataset[demo].attrs["model_file"]

    # Load XML from string
    root = ET.fromstring(model_file)
    tree = ET.ElementTree(root)

    ep_dir = extras_dir / f"episode_{demo_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Save compressed states
    np.savez_compressed(
        ep_dir / "states.npz",
        states=states,
    )

    # Save ep_meta
    with open(ep_dir / "ep_meta.json", "w") as f:
        json.dump(ep_meta, f, indent=4)

    # Pretty-print XML
    ET.indent(tree, space="  ")

    # Save XML as gzip
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
        f.write(xml_bytes)


def save_dataset_meta(dataset_dir: Path, hdf5_dataset) -> None:
    """
    Save dataset-level metadata from HDF5 dataset to JSON file. This information is
    needed for environment creation and not included in main lerobot dataset.
    """
    extras_dir = dataset_dir / "extras"
    dataset_meta_path = extras_dir / "dataset_meta.json"
    dataset_meta = dict(hdf5_dataset["data"].attrs)
    dataset_meta["total"] = int(dataset_meta["total"])
    dataset_meta["env_args"] = json.loads(dataset_meta["env_args"])
    if "env_info" in dataset_meta:
        dataset_meta["env_info"] = json.loads(dataset_meta["env_info"])
    with open(dataset_meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=4)


def get_env_metadata(dataset_dir: Path) -> dict:
    """
    Get environment metadata from dataset metadata JSON file.
    """
    extras_dir = dataset_dir / "extras"
    dataset_meta_path = extras_dir / "dataset_meta.json"
    with open(dataset_meta_path, "r") as f:
        dataset_meta = json.load(f)
    return dataset_meta["env_args"]


def get_modality_dict(dataset_dir: Path) -> dict:
    """
    Get modality dictionary from modality JSON file in dataset metadata directory.
    """
    lerobot_meta_dir = dataset_dir / "meta"
    modality_path = lerobot_meta_dir / "modality.json"
    with open(modality_path, "r") as f:
        modality_dict = json.load(f)
    return modality_dict


def get_episodes(dataset_dir: Path) -> list[Path]:
    """
    Get list of episode directories from dataset extras directory.
    """
    extras_dir = dataset_dir / "extras"
    episode_dirs = sorted(extras_dir.glob("episode_*"))
    return episode_dirs


def get_episode_states(dataset_dir: Path, ep_num) -> np.ndarray:
    """
    Get the states array for a given episode number.
    """
    episode_dir = dataset_dir / "extras" / f"episode_{ep_num:06d}"
    states_path = episode_dir / "states.npz"
    states_data = np.load(states_path)
    return states_data["states"]


def get_episode_model_xml(dataset_dir: Path, ep_num) -> str:
    """
    Get the model XML string for a given episode number.
    """
    episode_dir = dataset_dir / "extras" / f"episode_{ep_num:06d}"
    model_xml_path = episode_dir / "model.xml.gz"
    with gzip.open(model_xml_path, "rb") as f:
        xml_bytes = f.read()
    return xml_bytes.decode("utf-8")


def get_episode_meta(dataset_dir: Path, ep_num) -> dict:
    """
    Get the episode metadata for a given episode number.
    """
    episode_dir = dataset_dir / "extras" / f"episode_{ep_num:06d}"
    ep_meta_path = episode_dir / "ep_meta.json"
    with open(ep_meta_path, "r") as f:
        ep_meta = json.load(f)
    return ep_meta


def reorder_lerobot_action(action_lerobot: np.ndarray, dataset) -> np.ndarray:
    """
    Reorder lerobot action array to match HDF5 dataset ordering. Inverse of function reorder_hdf5_action.
    """
    modality_dict = get_modality_dict(dataset)
    action_info = modality_dict["action"]
    # sort keys based on start index in action info
    sorted_action_keys = sorted(
        action_info.keys(), key=lambda k: action_info[k]["start"]
    )
    reordered_action = np.zeros_like(action_lerobot)
    for key in sorted_action_keys:
        lerobot_start = action_info[key]["start"]
        lerobot_end = action_info[key]["end"]
        hdf5_start, hdf5_end = ACTION_KEY_ORDERING_HDF5[key]
        reordered_action[:, hdf5_start:hdf5_end] = action_lerobot[
            :, lerobot_start:lerobot_end
        ]
    return reordered_action


def get_episode_actions(dataset_dir: Path, ep_num, abs_actions=False) -> np.ndarray:
    """
    Get the actions array for a given episode number.
    """
    data_files = list(dataset_dir.glob(f"data/*/episode_{ep_num:06d}.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet file found for episode {ep_num}")
    df = pd.read_parquet(data_files[0])
    if abs_actions:
        raise NotImplementedError("Absolute actions not implemented yet")

    return reorder_lerobot_action(np.stack(df["action"].to_list()), dataset_dir)
