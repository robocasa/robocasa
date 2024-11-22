import os
import json
import argparse
import glob
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

import robocasa.utils.model_zoo.file_utils as FileUtils

from robosuite import load_controller_config
from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv

# from robomimic.envs.env_robosuite import EnvRobosuite


def play_object_montage(args):
    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        args.render_image_names = ["agentview"]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    subfolders = glob.glob(os.path.join(args.folder + "/", "*/" * args.subfolder_depth))

    for subf in subfolders:
        mjcf_paths = []
        for root, _, files in os.walk(subf):
            if "model.xml" in files:
                mjcf_paths.append(os.path.join(str(root), "model.xml"))
        mjcf_paths = mjcf_paths[: args.max_objects]

        # maybe dump video
        video_path = os.path.join(subf, "{}.mp4".format(FileUtils.get_stem(subf)))
        if os.path.exists(video_path) and not args.force_regenerate:
            print("Video already exists at:", video_path + ", skipping")
            continue
        print("Video saved at:", video_path)
        video_writer = imageio.get_writer(video_path, fps=20)

        env_kwargs = {}
        if args.no_noise:
            env_kwargs.update(
                x_range=(0.0, 0.0),
                y_range=(0.0, 0.0),
                rotation=(np.pi, np.pi),
                initialization_noise=None,
            )

        for mjcf_path in tqdm(mjcf_paths):
            # Create environment
            try:
                env = ObjectPlayEnv(
                    robots="Panda",
                    controller_configs=controller_config,
                    obj_mjcf_path=mjcf_path,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    render_camera=args.render_image_names,
                    **env_kwargs
                )
            except:
                continue

            model_name = mjcf_path.split("/")[-2]

            for i in range(args.n_resets):
                env.reset()
                # on-screen render
                if args.render:
                    env.render(mode="human", camera_name=args.render_image_names[0])

                # video render
                video_img = []
                for cam_name in args.render_image_names:
                    im = env.sim.render(
                        height=args.imsize, width=args.imsize, camera_name=cam_name
                    )[::-1]
                    if not args.no_text:
                        im = write_text_on_image(im, model_name)
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                for _ in range(args.n_steps):
                    video_writer.append_data(video_img)

            del env

        video_writer.close()

    os._exit(0)


def write_text_on_image(image_arr, text):
    img = Image.fromarray(image_arr)
    image_edit = ImageDraw.Draw(img)
    imsize = image_arr.shape[0]
    font = ImageFont.truetype(
        "objaverse/assets/JetBrainsMono-Medium.ttf", int(30 * imsize / 512)
    )
    image_edit.text((15, 15), text, (0, 0, 0), font=font, stroke_width=1)
    return np.array(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to model(s)",
    )

    parser.add_argument(
        "--subfolder_depth",
        type=int,
        default=0,
        help="generate a video for subfolder at level of depths",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=10,
        help="render n frames to video",
    )

    parser.add_argument(
        "--n_resets",
        type=int,
        default=3,
        help="number of resets per object",
    )

    parser.add_argument(
        "--no_noise",
        action="store_true",
        help="no initial placement noise",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # image size (for off-screen rendering with use-obs is False)
    parser.add_argument(
        "--imsize",
        type=int,
        default=1024,
        help="video image size (if applicable)",
    )

    # Whether to disable writing text to image
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="don't write model name on text",
    )

    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="the max number of objects to render (per category)",
    )

    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="regenerate video even if a video file is already present",
    )

    args = parser.parse_args()
    play_object_montage(args)
