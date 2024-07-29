import argparse
import os
from pathlib import Path
from termcolor import colored

import robocasa
from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
    get_ds_path
)
from robocasa.scripts.download_kitchen_assets import download_url

def download_datasets(tasks, ds_types, overwrite=False):
    if tasks is None:
        tasks = list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(MULTI_STAGE_TASK_DATASETS.keys())
    
    for task_name in tasks:
        for ds_type in ds_types:
            ds_path, ds_info = get_ds_path(task_name, ds_type, return_info=True)
            if ds_path is None:
                print("Skipping dataset type", ds_type, "for task", task_name, "since it is not available.")
                continue
            ds_dir = "/".join(ds_path.split("/")[0:-1])
            fname = ds_path.split("/")[-1]

            Path(ds_dir).mkdir(parents=True, exist_ok=True)

            print(colored(f"Task: {task_name}\nDataset type: {ds_type}", "yellow"))

            if overwrite is False and os.path.exists(ds_path):
                print(colored(f"Dataset already exists under {ds_path}\nSkipping.", "yellow"))
                print()
                continue

            download_url(
                url=ds_info["url"],
                download_dir=ds_dir,
                fname=fname,
                check_overwrite=(overwrite is False),
            )
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=None,
        help="Tasks to download datasets for. Defaults to all tasks"
    )

    parser.add_argument(
        "--ds_types",
        type=str,
        nargs='+',
        default=["human_raw", "human_im"],
        help="Dataset types. Choose one or multiple options among human_raw, human_im, mg_im",
    )

    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="automatically overwrite any existing files",
    )

    args = parser.parse_args()

    ans = input("This script may download several Gb of data. Proceed? (y/n) ")
    if ans == "y":
        print("Proceeding...")
    else:
        print("Aborting.")
        exit()

    download_datasets(
        tasks=args.tasks,
        ds_types=args.ds_types,
        overwrite=args.overwrite,
    )