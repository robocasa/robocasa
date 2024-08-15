import argparse
import json
import os
import time
from collections import OrderedDict

import robosuite
from termcolor import colored

import robocasa
from robocasa.scripts.download_datasets import download_datasets
from robocasa.scripts.download_kitchen_assets import download_and_extract_zip
from robocasa.scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry import get_ds_path
import os


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, help="task (must be task with demos collected already)"
    )
    parser.add_argument(
        "--render_offscreen",
        action="store_true",
        help="off-screen rendering",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/robocasa_demo_tasks",
        help="path to video folder for offscreen rendering.",
    )
    args = parser.parse_args()

    tasks = OrderedDict(
        [
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )

    video_num = -1
    while True:
        if args.task is None:
            task = choose_option(
                tasks, "task", default="PnPCounterToCab", show_keys=True
            )
        else:
            task = args.task
        video_num += 1

        dataset = get_ds_path(task, ds_type="human_raw")

        if os.path.exists(dataset) is False:
            # download dataset files
            print(
                colored(
                    "Unable to find dataset locally. Downloading...", color="yellow"
                )
            )
            download_datasets(tasks=[task], ds_types=["human_raw"])

        parser = argparse.Namespace()
        parser.dataset = dataset

        if args.render_offscreen:
            parser.render = True
            if not os.path.exists(args.video_path):
                os.makedirs(args.video_path)
            parser.video_path = os.path.join(args.video_path, f"video_{video_num}.mp4")
        else:
            parser.render = False
            parser.video_path = False

        parser.render = not args.render_offscreen
        parser.use_actions = False
        parser.render_image_names = ["robot0_agentview_center"]
        parser.use_obs = False
        parser.n = 1 if args.task is None else None
        parser.filter_key = None
        parser.video_skip = 5
        parser.first = False
        parser.verbose = True
        parser.extend_states = True

        playback_dataset(parser)
        if args.task is not None:
            break
        print()
