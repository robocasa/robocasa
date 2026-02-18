import argparse
import os
import tarfile
from pathlib import Path

from termcolor import colored

import robocasa
from robocasa.utils.dataset_registry import (
    COMPOSITE_TASK_DATASETS,
    ATOMIC_TASK_DATASETS,
)
from robocasa.macros import DATASET_BASE_PATH
from robocasa.utils.dataset_registry_utils import get_ds_meta
from huggingface_hub import hf_hub_download


def download_datasets(
    split, tasks, source, all_data=False, overwrite=False, dryrun=False
):

    if all_data:
        tasks = None
        source = ["human", "mimicgen"]
        split = ["pretrain", "target"]

    if tasks is None:
        tasks = list(ATOMIC_TASK_DATASETS.keys()) + list(COMPOSITE_TASK_DATASETS.keys())

    for task_name in tasks:
        for sp in split:
            for src in source:
                print(
                    colored(f"Task: {task_name}\nSplit: {sp}\nSource: {src}", "yellow")
                )

                ds_meta = get_ds_meta(
                    task=task_name, source="mg" if src == "mimicgen" else src, split=sp
                )  # mg means mimicgen
                ds_path = ds_meta["path"] if ds_meta is not None else None
                if ds_path is None:
                    print(
                        colored(
                            f"No dataset for this task and dataset type exists!\nSkipping.\n",
                            "yellow",
                        )
                    )
                    continue

                if overwrite is False and os.path.exists(ds_path):
                    print(
                        colored(
                            f"Dataset already exists under {ds_path}\nSkipping.\n",
                            "yellow",
                        )
                    )
                    continue

                if not dryrun:
                    download_and_extract_from_hf(
                        destination=ds_path,
                    )
                else:
                    print(colored(f"Would download to {ds_path}", "cyan"))
                print()


def download_and_extract_from_hf(destination):
    """Download a tar file from HF and extract it to the destination path."""
    ds_path = Path(destination)
    if DATASET_BASE_PATH is not None:
        base_datasets_path = Path(DATASET_BASE_PATH) / "v1.0"
    else:
        # Get the robocasa package path
        robocasa_path = robocasa.__path__[0]
        # Construct the datasets path
        base_datasets_path = Path(robocasa_path).parent / "datasets" / "v1.0"
    relative_path = ds_path.relative_to(base_datasets_path)
    tar_filename = str(relative_path.parent / f"{relative_path.name}.tar")

    print(colored(f"Downloading {tar_filename}...", "green"))

    # Download the single tar file
    tar_path = hf_hub_download(
        repo_id="nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos",
        repo_type="dataset",
        filename=tar_filename,
        revision="main",
    )
    extract_dir = ds_path.parent
    os.makedirs(extract_dir, exist_ok=True)

    print(colored(f"Extracting to {extract_dir}...", "green"))
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)

    # clean up tar path to prevent cache from blowing up
    os.remove(tar_path)

    print(colored(f"Done: {ds_path}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to download datasets for. Defaults to all tasks",
    )
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        default=["human"],
        choices=["human", "mimicgen"],
        help="Dataset types. Choose one or multiple options among human, mimicgen, all",
    )

    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["target"],
        choices=["pretrain", "target"],
        help="Dataset splits to download. Choose one or multiple options among pretrain, target, all",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="automatically overwrite any existing files",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="simulate without downloading datasets",
    )

    args = parser.parse_args()

    ans = input("This script may download several Gb of data. Proceed? (y/n) ")
    if ans == "y":
        print("Proceeding...")
    else:
        print("Aborting.")
        exit()

    download_datasets(
        all_data=args.all,
        split=args.split,
        tasks=args.tasks,
        source=args.source,
        overwrite=args.overwrite,
        dryrun=args.dryrun,
    )
