from tqdm import tqdm
import os
from pathlib import Path
import urllib.request
import time

import robocasa
from zipfile import ZipFile

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
    request.get_method = lambda: 'HEAD'

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
        user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
        assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)

def download_and_extract_zip(url, folder, check_folder_exists=True):
    assert url.endswith(".zip")

    download_dir = os.path.abspath(os.path.join(folder, os.pardir))
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    download_path = os.path.join(
        download_dir,
        "{}.zip".format(os.path.basename(folder))
    )
    
    # extract files
    if check_folder_exists and os.path.exists(folder):
        ans = input("{} already exists! \noverwrite? (y/n)\n".format(folder))

        if ans == "y":
            print("Proceeding, downloading to {}".format(folder))
        else:
            print("Skipping.\n")
            return
        
    print("Downloading...")
    
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
            print("Error downloading after try #{}".format(i+1))

    if download_success is False:
        print("Failed to download. Try again...")
        return
        
    print("Extracting...")
    with ZipFile(download_path, "r") as zip_ref:
        # Extracting all the members of the zip 
        # into a specific location.
        zip_ref.extractall(path=download_dir)

    # delete zip file
    os.remove(download_path)

    print("Done.\n")


def download_kitchen_assets():
    print("Downloading environment textures")
    download_and_extract_zip(
        url="https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/textures"),
        check_folder_exists=False,
    )
    
    print("Downloading fixtures")
    download_and_extract_zip(
        url="https://utexas.box.com/shared/static/956d0w2ucqs7d3eors1idsohgum57nli.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        check_folder_exists=False,
    )

    print("Downloading objaverse objects")
    download_and_extract_zip(
        url="https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse"),
        check_folder_exists=False,
    )

    print("Downloading AI-generated objects")
    download_and_extract_zip(
        url="https://utexas.box.com/shared/static/89i7lqpngmnudfskh4xztri2yuiuf3g2.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/aigen_objs"),
        check_folder_exists=False,
    )

    print("Downloading generative textures")
    download_and_extract_zip(
        url="https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/generative_textures"),
        check_folder_exists=False,
    )

if __name__ == "__main__":
    download_kitchen_assets()