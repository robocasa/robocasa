import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.scripts.collect_demos import collect_human_trajectory


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
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="task")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    parser.add_argument("--robot", type=str, help="robot")
    args = parser.parse_args()

    robots = OrderedDict([(0, "PandaMobile"), (1, "GR1FloatingBody")])

    raw_layouts = dict(
        map(lambda item: (item.value, item.name.lower().capitalize()), LayoutType)
    )
    layouts = OrderedDict()
    for k in sorted(raw_layouts.keys()):
        if k < -0:
            continue
        layouts[k] = raw_layouts[k]

    raw_styles = dict(
        map(lambda item: (item.value, item.name.lower().capitalize()), StyleType)
    )
    styles = OrderedDict()
    for k in sorted(raw_styles.keys()):
        if k < 0:
            continue
        styles[k] = raw_styles[k]

    if args.robot is None:
        robot_choice = choose_option(
            robots, "robot", default=0, default_message="PandaMobile"
        )
        robot = robots[robot_choice]
    else:
        robot = args.robot

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": robot,
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
    }

    args.renderer = "mjviewer"

    print(colored("Initializing environment...", "yellow"))

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)

    # collect demonstrations
    while True:
        if args.layout is None:
            layout = choose_option(
                layouts, "kitchen layout", default=-1, default_message="random layouts"
            )
        else:
            layout = args.layout

        if args.style is None:
            style = choose_option(
                styles, "kitchen style", default=-1, default_message="random styles"
            )
        else:
            style = args.style

        if layout == -1:
            layout = np.random.choice(range(10))
        if style == -1:
            style = np.random.choice(range(11))

        env.layout_and_style_ids = [[layout, style]]
        print(
            colored(
                f"Showing configuration:\n    Layout: {layouts[layout]}\n    Style: {styles[style]}",
                "green",
            )
        )
        print()
        print(
            colored(
                "Spawning environment...\n(Press Q any time to view new configuration)",
                "yellow",
            )
        )

        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(args.renderer != "mjviewer"),
            max_fr=30,
            print_info=False,
        )

        print()
        print()
