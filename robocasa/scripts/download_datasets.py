import argparse
import json
import os
import tarfile
import urllib.request
from pathlib import Path

from termcolor import colored
from tqdm import tqdm

import robocasa
from robocasa.utils.dataset_registry import (
    COMPOSITE_TASK_DATASETS,
    ATOMIC_TASK_DATASETS,
)
from robocasa.macros import DATASET_BASE_PATH
from robocasa.utils.dataset_registry_utils import get_ds_meta

# path to the box_links_ds.json shipped with robocasa
BOX_LINKS_DS_PATH = os.path.join(
    robocasa.__path__[0], "models", "assets", "box_links", "box_links_ds.json"
)
with open(BOX_LINKS_DS_PATH, "r") as f:
    BOX_LINKS_DS = json.load(f)


def _get_direct_download_url(shared_url, ext="tar"):
    """
    Convert a Box shared link into a direct-download URL.
    e.g. https://utexas.box.com/s/abc123  →
         https://utexas.box.com/shared/static/abc123.tar
    """
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, download_dir, fname=None, check_overwrite=True):
    """
    Downloads the file at the given url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.
    """
    if fname is None:
        fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n "
        )
        assert user_response.lower() in {
            "yes",
            "y",
        }, "Did not receive confirmation. Aborting download."

    print(colored(f"Downloading to {file_to_write}", "yellow"))

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)


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
                    download_and_extract_from_box(
                        destination=ds_path,
                    )
                else:
                    print(colored(f"Would download to {ds_path}", "cyan"))
                print()


def download_and_extract_from_box(destination):
    """Download a tar file from Box and extract it to the destination path."""
    ds_path = Path(destination)
    if DATASET_BASE_PATH is not None:
        base_datasets_path = Path(DATASET_BASE_PATH) / "v1.0"
    else:
        robocasa_path = robocasa.__path__[0]
        base_datasets_path = Path(robocasa_path).parent / "datasets" / "v1.0"
    relative_path = ds_path.relative_to(base_datasets_path)
    # box_links_ds.json keys look like: pretrain/atomic/TaskName/20250820/lerobot.tar
    tar_key = (relative_path.parent / f"{relative_path.name}.tar").as_posix()

    if tar_key not in BOX_LINKS_DS:
        print(colored(f"No Box link found for '{tar_key}' - skipping.", "red"))
        return

    shared_url = BOX_LINKS_DS[tar_key]
    download_url_str = _get_direct_download_url(shared_url, ext="tar")

    extract_dir = ds_path.parent
    os.makedirs(extract_dir, exist_ok=True)

    tar_download_path = os.path.join(str(extract_dir), f"{relative_path.name}.tar")

    print(colored(f"Downloading {tar_key} from Box...", "green"))

    download_success = False
    for i in range(3):
        try:
            download_url(
                url=download_url_str,
                download_dir=str(extract_dir),
                fname=f"{relative_path.name}.tar",
                check_overwrite=False,
            )
            download_success = True
            break
        except Exception as e:
            print(f"Error downloading after try #{i + 1}: {e}")

    if not download_success:
        print("Failed to download. Try again...")
        if os.path.exists(tar_download_path):
            os.remove(tar_download_path)
        return

    print(colored(f"Extracting to {extract_dir}...", "green"))
    with tarfile.open(tar_download_path, "r") as tar:
        tar.extractall(path=extract_dir)

    # clean up tar file
    os.remove(tar_download_path)

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
