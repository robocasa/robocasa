import argparse
import json
import sys
import termios
from collections import OrderedDict

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
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
            print("[{}] {}: {}".format(i + 1, k, v))
        else:
            print("[{}] {}".format(i + 1, v))
    print()
    try:
        s = input(
            "Choose an option 1 to {}, or 0/other for default ({}): ".format(
                len(options),
                default_message,
            )
        )
        num = int(s)
        # 0 means use default (e.g. random layouts/styles)
        if num == 0:
            choice = default
        else:
            # parse 1-based input into 0-based index
            k = min(max(num - 1, 0), len(options) - 1)
            choice = list(options.keys())[k]
    except Exception as e:  # noqa: F841
        print(f"Invalid input: {e}")
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
    parser.add_argument("--task", type=str, default="Kitchen", help="task")
    parser.add_argument(
        "--layout", type=int, help="kitchen layout (choose number 1-60)"
    )
    parser.add_argument("--style", type=int, help="kitchen style (choose number 1-60)")
    parser.add_argument("--robot", type=str, help="robot", default="PandaOmron")
    args = parser.parse_args()

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

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
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
        print("chose layout:", layout, "style:", style)
        if layout == -1:
            layout = int(np.random.choice(list(layouts.keys())))
        if style == -1:
            style = int(np.random.choice(list(styles.keys())))

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

        # Flush stdin to clear any buffered keypresses otherwise when getting
        # the next layout/style choice, it will read a 'q' from before
        termios.tcflush(sys.stdin, termios.TCIFLUSH)

        print()
        print()
