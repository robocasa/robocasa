import argparse
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download
from termcolor import colored

import robocasa

ROBOCASA_ASSETS_HF_REPO = "robocasa/robocasa-assets"
LW_ASSETS_HF_REPO = "nvidia/PhysicalAI-Kitchen-Assets"

DOWNLOAD_ASSET_REGISTRY = {
    ### textures ###
    "tex": dict(
        message="Downloading environment textures",
        hf_repo=ROBOCASA_ASSETS_HF_REPO,
        hf_filename="textures.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/textures"),
        check_folder_exists=False,
    ),
    "tex_generative": dict(
        message="Downloading AI-generated environment textures",
        hf_repo=ROBOCASA_ASSETS_HF_REPO,
        hf_filename="generative_textures.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/generative_textures"),
        check_folder_exists=False,
    ),
    ### fixtures ###
    "fixtures_lw": dict(
        message="Downloading lightwheel fixtures",
        hf_repo=LW_ASSETS_HF_REPO,
        hf_filename="fixtures_lightwheel.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        check_folder_exists=False,
    ),
    ### objects ###
    "objs_objaverse": dict(
        message="Downloading objaverse objects",
        hf_repo=ROBOCASA_ASSETS_HF_REPO,
        hf_filename="objaverse.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse"),
        check_folder_exists=False,
    ),
    "objs_aigen": dict(
        message="Downloading AI-generated objects",
        hf_repo=ROBOCASA_ASSETS_HF_REPO,
        hf_filename="aigen_objs.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/aigen_objs"),
        check_folder_exists=False,
    ),
    "objs_lw": dict(
        message="Downloading lightwheel objects",
        hf_repo=LW_ASSETS_HF_REPO,
        hf_filename="objects_lightwheel.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/lightwheel"),
        check_folder_exists=False,
    ),
}


def download_and_extract_from_hf(
    hf_repo,
    hf_filename,
    folder,
    check_folder_exists=True,
    delete_old_folder=False,
    message="Downloading...",
):
    """Download a zip file from a HuggingFace repo and extract it."""
    download_dir = os.path.abspath(os.path.join(folder, os.pardir))
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    if delete_old_folder and os.path.exists(folder):
        print(colored(f"Deleting existing folder: {folder}", "yellow"))
        shutil.rmtree(folder)

    print(colored(message, "yellow"))

    # check if folder already exists
    if check_folder_exists and os.path.exists(folder):
        ans = input("{} already exists! \noverwrite? (y/n) ".format(folder))
        if ans == "y":
            print(colored("Proceeding.", "yellow"))
        else:
            print(colored("Skipping.\n", "yellow"))
            return

    download_success = False
    for i in range(3):
        try:
            print(colored(f"Downloading {hf_filename} from {hf_repo}...", "yellow"))
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                repo_type="dataset",
                filename=hf_filename,
                revision="main",
            )
            download_success = True
            break
        except Exception as e:
            print(f"Error downloading after try #{i + 1}: {e}")

    if not download_success:
        print("Failed to download. Try again...")
        return

    print(colored("Extracting...", "yellow"))
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path=download_dir)

    os.remove(zip_path)

    print(colored("Done.\n", "yellow"))


def download_kitchen_assets(types):
    ans = input("The script will download ~10 Gb of data. Proceed? (y/n) ")
    if ans == "y":
        print("Proceeding...")
    else:
        print("Aborting.")
        return

    for ds_name, config in DOWNLOAD_ASSET_REGISTRY.items():
        if types is None:
            pass
        elif "all" in types:
            # download everything
            pass
        else:
            if ds_name not in types:
                continue
        download_and_extract_from_hf(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        nargs="+",
        choices=list(DOWNLOAD_ASSET_REGISTRY.keys()) + ["all"],
        help='asset registry types to download (specify "all" to download all types)',
    )

    args = parser.parse_args()
    types = args.type

    download_kitchen_assets(types)
