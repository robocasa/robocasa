import argparse
import json
import time
from collections import OrderedDict
import os

import robocasa
import robosuite
from termcolor import colored
from robocasa.scripts.playback_dataset import playback_dataset
from robocasa.scripts.download_kitchen_assets import download_and_extract_zip

def choose_option(options, option_name, show_keys=False, default=None, default_message=None):
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
        s = input("Choose an option 0 to {}, or any other key for default ({}): ".format(
            len(options) - 1,
            default_message,
        ))
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
    parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
    args = parser.parse_args()


    tasks = OrderedDict([
        ("PnPCounterToCab", "pick and place from counter to cabinet"),
        ("PnPCounterToSink", "pick and place from counter to sink"),
        ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
        ("PnPStoveToCounter", "pick and place from stove to counter"),
        ("OpenSingleDoor", "open cabinet or microwave door"),
        ("CloseDrawer", "close drawer"),
        ("TurnOnMicrowave", "turn on microwave"),
        ("TurnOnSinkFaucet", "turn on sink faucet"),
        ("TurnOnStove", "turn on stove"),
        ("PrepareCoffee", "make coffee"),
        ("PreSoakPan", "prepare pan for washing"),
        ("RestockPantry", "restock cans in pantry"),
    ])

    while True:
        if args.task is None:
            task = choose_option(tasks, "task", default="PnPCounterToCab", show_keys=True)

        if task == "PnPCounterToCab":
            dataset = "single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo.hdf5"
        elif task == "PnPCounterToSink":
            dataset = "single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo.hdf5"
        elif task == "PnPMicrowaveToCounter":
            dataset = "single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo.hdf5"
        elif task == "PnPStoveToCounter":
            dataset = "single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo.hdf5"
        elif task == "OpenSingleDoor":
            dataset = "single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo.hdf5"
        elif task == "CloseDrawer":
            dataset = "single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5"
        elif task == "TurnOnMicrowave":
            dataset = "single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo.hdf5"
        elif task == "TurnOnSinkFaucet":
            dataset = "single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo.hdf5"
        elif task == "TurnOnStove":
            dataset = "single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo.hdf5"
        elif task == "PrepareCoffee":
            dataset = "multi_stage/brewing/PrepareCoffee/2024-05-07/demo.hdf5"
        elif task == "PreSoakPan":
            dataset = "multi_stage/washing_dishes/PreSoakPan/2024-05-10/demo.hdf5"
        elif task == "RestockPantry":
            dataset = "multi_stage/restocking_supplies/RestockPantry/2024-05-10/demo.hdf5"

        dataset = os.path.join(os.path.dirname(robocasa.__file__), "models/assets/datasets", dataset)

        if os.path.exists(dataset) is False:
            # download dataset files
            print("Unable to find dataset locally. Downloading dataset files.")
            if "single_stage" in dataset:
                download_and_extract_zip(
                    url="https://utexas.box.com/shared/static/kcl93ptkp8w8gxj7el48d1hz6qve4igr.zip",
                    folder=os.path.join(robocasa.__path__[0], "models/assets/datasets/single_stage"),
                    check_folder_exists=False,
                )
            elif "multi_stage" in dataset:
                download_and_extract_zip(
                    url="https://utexas.box.com/shared/static/x7o3u6d278x6nohyhqw5qe00qq9hon6w.zip",
                    folder=os.path.join(robocasa.__path__[0], "models/assets/datasets/multi_stage"),
                    check_folder_exists=False,
                )

        parser = argparse.ArgumentParser()
        parser.dataset = dataset
        parser.video_path = None
        parser.render = True
        parser.use_actions = False
        parser.render_image_names = ["robot0_agentview_center"]
        parser.use_obs = False
        parser.n = 1
        parser.filter_key = None
        parser.video_skip = 5
        parser.first = False

        print(colored(f"Playing trajectory for {task}", "yellow"))

        playback_dataset(parser)

        print()
        print()
