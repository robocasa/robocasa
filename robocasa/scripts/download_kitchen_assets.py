import argparse
import os
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from termcolor import colored
from tqdm import tqdm

import robocasa

DOWNLOAD_ASSET_REGISTRY = {
    ### textures ###
    "tex": dict(
        message="Downloading environment textures",
        url="https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/textures"),
        check_folder_exists=False,
    ),
    "tex_generative": dict(
        message="Downloading AI-generated environment textures",
        url="https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/generative_textures"),
        check_folder_exists=False,
    ),
    ### fixtures ###
    "fixtures_v0.1": dict(
        message="Downloading RoboCasa v0.1 fixtures",
        url="https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        check_folder_exists=False,
    ),
    "fixtures_lw": dict(
        message="Downloading lightwheel fixtures",
        url="https://utexas.box.com/shared/static/51njmsj373pzzqrpl5yu2o5gox73mqt9.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        check_folder_exists=False,
    ),
    ### objects ###
    "objs_objaverse": dict(
        message="Downloading objaverse objects",
        url="https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse"),
        check_folder_exists=False,
    ),
    "objs_aigen": dict(
        message="Downloading AI-generated objects",
        url="https://utexas.box.com/shared/static/os3hrui06lasnuvwqpmwn0wcrduh6jg3.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/aigen_objs"),
        check_folder_exists=False,
    ),
    "objs_lw": dict(
        message="Downloading lightwheel objects",
        url="https://utexas.box.com/shared/static/xhbxldptvssy5ibbsxogv42xhnajcp8t.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/lightwheel"),
        check_folder_exists=False,
    ),
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.

    Args:
        url (str): url string

    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, fname=None, check_overwrite=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.

    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    # assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    # time.sleep(0.5)

    if fname is None:
        # infer filename from url link
        fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n "
        )
        assert user_response.lower() in {
            "yes",
            "y",
        }, f"Did not receive confirmation. Aborting download."

    print(colored(f"Downloading to {file_to_write}", "yellow"))

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)


def download_and_extract_zip(
    url,
    folder,
    check_folder_exists=True,
    prompt_before_download=False,
    message="Downloading...",
):
    assert url.endswith(".zip")

    download_dir = os.path.abspath(os.path.join(folder, os.pardir))
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    download_path = os.path.join(
        download_dir, "{}.zip".format(os.path.basename(folder))
    )

    print(colored(message, "yellow"))

    # extract files
    if check_folder_exists and os.path.exists(folder):
        ans = input("{} already exists! \noverwrite? (y/n) ".format(folder))

        if ans == "y":
            print(colored("Proceeding.", "yellow"))
        else:
            print(colored("Skipping.\n", "yellow"))
            return

    if prompt_before_download:
        ans = input(
            "Assets to be downloaded may be a few Gb. Proceed? (y/n) ".format(folder)
        )

        if ans == "y":
            print(colored("Proceeding.", "yellow"))
        else:
            print(colored("Skipping.\n", "yellow"))
            return

    download_success = False
    for i in range(3):
        try:
            download_url(
                url=url,
                download_dir=download_dir,
                fname=os.path.basename(download_path),
                check_overwrite=False,
            )
            download_success = True
            break
        except:
            print("Error downloading after try #{}".format(i + 1))

    if download_success is False:
        print("Failed to download. Try again...")
        return

    print(colored("Extracting...", "yellow"))
    with ZipFile(download_path, "r") as zip_ref:
        # Extracting all the members of the zip
        # into a specific location.
        ret = zip_ref.extractall(path=download_dir)

    # delete zip file
    os.remove(download_path)

    print(colored("Done.\n", "yellow"))


def download_kitchen_assets(types):
    ans = input("The script will download ~5 Gb of data. Proceed? (y/n) ")
    if ans == "y":
        print("Proceeding...")
    else:
        print("Aborting.")
        return

    for ds_name, config in DOWNLOAD_ASSET_REGISTRY.items():
        if types is None:
            if ds_name == "aigen_objs":
                continue  # skip for now, too large to download initially
        elif "all" in types:
            # download everything
            pass
        else:
            if ds_name not in types:
                continue
        download_and_extract_zip(**config)


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
