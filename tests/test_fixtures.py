import os
import numpy as np
import tqdm
import traceback
import argparse

from robocasa.scripts.browse_mjcf_model import read_model
import robocasa.macros as macros
from robocasa.models.fixtures.fixture import FixtureType

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


def get_all_dict_items(dictionary):
    items = []
    for key, value in dictionary.items():
        inner_dicts = []
        if isinstance(value, dict):
            inner_dicts.append(value)
        elif isinstance(value, list):
            for elem in value:
                if isinstance(elem, dict):
                    inner_dicts.append(elem)
        else:
            items.append((key, value))
        for d in inner_dicts:
            items.extend(get_all_dict_items(d))
    return items


def get_layout_ids(filter_by=None):
    layout_id_list = []
    for i in range(10):
        layout_path = get_layout_path(layout_id=i)
        if filter_by is not None:
            with open(layout_path, "r") as f:
                layout_config = yaml.safe_load(f)
            layout_items = get_all_dict_items(layout_config)
            if filter_by not in layout_items:
                continue
        layout_id_list.append(i)
    return layout_id_list


FIXTURE_TO_TEST_ENVS = dict(
    microwave=[
        dict(env_name="PnPCounterToMicrowave"),
        dict(env_name="PnPMicrowaveToCounter"),
        dict(env_name="OpenSingleDoor", door_id=FixtureType.MICROWAVE),
        dict(env_name="CloseSingleDoor", door_id=FixtureType.MICROWAVE),
        dict(env_name="TurnOnMicrowave"),
        dict(env_name="TurnOffMicrowave"),
    ],
    stove=[
        dict(env_name="PnPCounterToStove"),
        dict(env_name="PnPStoveToCounter"),
        dict(env_name="TurnOnStove"),
        dict(env_name="TurnOffStove"),
    ],
    stovetop=[
        dict(env_name="PnPCounterToStove"),
        dict(env_name="PnPStoveToCounter"),
        dict(env_name="TurnOnStove"),
        dict(env_name="TurnOffStove"),
    ],
    sink=[
        dict(env_name="PnPCounterToSink"),
        dict(env_name="PnPSinkToCounter"),
        dict(env_name="TurnOnSinkFaucet"),
        dict(env_name="TurnOffSinkFaucet"),
        dict(env_name="TurnSinkSpout"),
    ],
    coffee_machine=[
        dict(env_name="CoffeeSetupMug"),
        dict(env_name="CoffeeServeMug"),
        dict(env_name="CoffeePressButton"),
    ],
    dishwasher=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.DISHWASHER),
    ],
    oven=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.OVEN),
    ],
    fridge_french_door=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.FRIDGE),
    ],
    fridge_side_by_side=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.FRIDGE),
    ],
    fridge_bottom_freezer=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.FRIDGE),
    ],
    toaster=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.TOASTER),
    ],
    toaster_oven=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.TOASTER_OVEN),
    ],
    blender=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.BLENDER),
    ],
    stand_mixer=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.STAND_MIXER),
    ],
    electric_kettle=[
        dict(env_name="Kitchen", init_robot_base_pos=FixtureType.ELECTRIC_KETTLE),
    ],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    fixture_type_list = list(FIXTURE_TO_TEST_ENVS.keys())
    for fixture_type in fixture_type_list:
        parser.add_argument(f"--{fixture_type}", type=str, nargs="+")

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

    for fixture_type in fixture_type_list:
        fixture_list = eval(f"args.{fixture_type}")

        if fixture_list is None:
            continue

        env_kwargs_list = FIXTURE_TO_TEST_ENVS[fixture_type]

        device = None

        # get valid layouts for this fixture type
        layout_ids = get_layout_ids(filter_by=("type", fixture_type))

        for fixture_name in fixture_list:
            style_configs = get_all_style_configs()
            for cfg in style_configs:
                cfg[fixture_type] = fixture_name

            for env_kwargs in env_kwargs_list:
                env = create_env(
                    render_onscreen=args.interactive,
                    seed=0,  # set seed=None to run unseeded
                    style_ids=style_configs,
                    layout_ids=layout_ids,
                    translucent_robot=True,
                    **env_kwargs,
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
