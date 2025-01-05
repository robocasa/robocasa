import os
import numpy as np
import tqdm
import traceback
import argparse

from robocasa.scripts.browse_mjcf_model import read_model
import robocasa
import robocasa.macros as macros

from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.models.scenes.scene_registry import get_layout_path, get_style_path
import yaml


def get_all_style_configs():
    style_config_list = []
    for i in range(12):
        style_path = get_style_path(style_id=i)
        with open(style_path, "r") as f:
            style_config = yaml.safe_load(f)
        style_config_list.append(style_config)
    return style_config_list


FIXTURE_TO_TEST_ENVS = dict(
    microwave=[
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
    ],
    stove=["PnPCounterToStove", "PnPStoveToCounter", "TurnOnStove", "TurnOffStove"],
    sink=[
        "PnPCounterToSink",
        "PnPSinkToCounter",
        "TurnOnSinkFaucet",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
    ],
    coffee_machine=["CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton"],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--microwave", type=str, nargs="+")
    parser.add_argument("--stove", type=str, nargs="+")
    parser.add_argument("--sink", type=str, nargs="+")
    parser.add_argument("--coffee_machine", type=str, nargs="+")

    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=["keyboard", "spacemouse"],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
    )

    args = parser.parse_args()

    for fixture_type in ["microwave", "stove", "sink", "coffee_machine"]:
        if fixture_type == "microwave":
            fixture_list = args.microwave
        elif fixture_type == "stove":
            fixture_list = args.stove
        elif fixture_type == "sink":
            fixture_list = args.sink
        elif fixture_type == "coffee_machine":
            fixture_list = args.coffee_machine

        if fixture_list is None:
            continue

        all_test_envs = FIXTURE_TO_TEST_ENVS[fixture_type]

        device = None

        for fixture_name in fixture_list:
            style_configs = get_all_style_configs()
            for cfg in style_configs:
                cfg[fixture_type] = fixture_name

            for test_env in all_test_envs:
                env = create_env(
                    env_name=test_env,
                    render_onscreen=args.interactive,
                    seed=0,  # set seed=None to run unseeded
                    style_ids=style_configs,
                )

                if args.interactive is False:
                    info = run_random_rollouts(
                        env,
                        num_rollouts=3,
                        num_steps=100,
                        video_path=f"/tmp/robocasa_test_fixtures/{fixture_type}_{fixture_name}_{test_env}.mp4",
                    )
                else:
                    # set up devices for interactive mode
                    if device is None:
                        if args.device == "keyboard":
                            from robosuite.devices import Keyboard

                            device = Keyboard(
                                env=env,
                                pos_sensitivity=args.pos_sensitivity,
                                rot_sensitivity=args.rot_sensitivity,
                            )
                        elif args.device == "spacemouse":
                            from robosuite.devices import SpaceMouse

                            device = SpaceMouse(
                                env=env,
                                pos_sensitivity=args.pos_sensitivity,
                                rot_sensitivity=args.rot_sensitivity,
                                vendor_id=macros.SPACEMOUSE_VENDOR_ID,
                                product_id=macros.SPACEMOUSE_PRODUCT_ID,
                            )
                        else:
                            raise ValueError

                    try:
                        while True:
                            collect_human_trajectory(
                                env,
                                device,
                                "right",
                                "single-arm-opposed",
                                True,
                                render=False,
                                max_fr=30,
                            )
                    except KeyboardInterrupt:
                        print("\n\nMoving onto next configuration...")

                env.close()
