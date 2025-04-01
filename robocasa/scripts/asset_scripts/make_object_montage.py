import argparse
import os

import imageio
from robosuite.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
from robosuite.scripts.browse_mjcf_model import get_model_screenshot
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_group", type=str, default="all")
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()

    mjcf_paths = []
    for cat_name in OBJ_GROUPS[args.obj_group]:
        cat = OBJ_CATEGORIES[cat_name]
        mjcf_paths += cat.mjcf_paths

    cam_settings = {
        "distance": 0.30,
        "elevation": -45,
    }

    video_path = os.path.expanduser(args.video_path)
    video_writer = imageio.get_writer(video_path, fps=5)

    for p in tqdm(mjcf_paths):
        try:
            image = get_model_screenshot(filepath=p, cam_settings=cam_settings)
            video_writer.append_data(image)
        except:
            pass

    video_writer.close()
