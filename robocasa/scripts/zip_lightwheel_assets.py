import shutil, errno
import os
import re

import robocasa


def copyanything(src, dst):
    """
    copied from https://stackoverflow.com/a/1994840
    """
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


def copy_lw_assets(src_dir, target_dir):
    for (root, dirs, files) in os.walk(src_dir):
        if "model.xml" not in files:
            continue
        asset_name = os.path.basename(root)
        pattern = r"^[a-zA-Z]+\d{3}$"
        is_lw_asset = bool(re.match(pattern, asset_name))
        if is_lw_asset:
            new_root = root.replace(src_dir, target_dir)
            copyanything(root, new_root)


src_dir = os.path.join(robocasa.__path__[0], "models/assets/fixtures")
target_dir = os.path.join(robocasa.__path__[0], "models/assets/fixtures_lw")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

print("preparing assets...")
copy_lw_assets(src_dir, target_dir)

# zip target dir
print("zipping...")
shutil.make_archive(target_dir, "zip", target_dir)  # zip assets
shutil.rmtree(target_dir)  # delete the temp folder
print(f"Wrote to {target_dir}.zip")
print("done.")
