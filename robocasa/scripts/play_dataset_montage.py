from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.scripts.playback_dataset import playback_dataset

import os
import argparse
import h5py
import json
from termcolor import colored

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="(optional) directory of datasets to play back. defaults to registered human datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to store videos",
        default="/tmp/robocasa_dataset_montage",
    )
    parser.add_argument(
        "--num_demos_per_task",
        type=int,
        help="number of demos to play per task",
        default=10,
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir = os.path.expanduser(output_dir)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    if args.directory is not None:
        # search for paths
        ds_paths = [
            "/Users/soroushnasiriany/Downloads/composite_task_sample_demos/2024-12-24-18-20-34_StackBowlsInSink/demo.hdf5",
            "/Users/soroushnasiriany/Downloads/composite_task_sample_demos/2024-12-24-18-16-29_PrewashFoodAssembly/demo.hdf5",
        ]
    else:
        ds_paths = [
            get_ds_path(task, ds_type="human_raw")
            for task in list(SINGLE_STAGE_TASK_DATASETS)
            + list(MULTI_STAGE_TASK_DATASETS)
        ]

    for ds_i, ds_path in enumerate(ds_paths):
        # infer task name
        f = h5py.File(ds_path)
        task = json.loads(f["data"].attrs["env_args"])["env_name"]
        f.close()

        parser = argparse.Namespace()
        parser.dataset = ds_path

        parser.render = False
        parser.video_path = os.path.join(output_dir, f"{task}.mp4")
        parser.use_actions = False
        parser.use_abs_actions = False
        parser.render_image_names = ["robot0_agentview_center", "robot0_eye_in_hand"]
        parser.use_obs = False
        parser.n = args.num_demos_per_task
        parser.filter_key = None
        parser.video_skip = 5
        parser.first = False
        parser.verbose = False
        parser.extend_states = True
        parser.camera_height = 512
        parser.camera_width = 768

        print(
            colored(
                f"[{ds_i+1} / {len(ds_paths)}] Playing sample demos for {task}",
                "yellow",
            )
        )
        playback_dataset(parser)
        print()
        print()
