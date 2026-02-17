import os
import hid
import numpy as np
import tqdm
import traceback
import argparse
import yaml
from datetime import datetime
from termcolor import colored

import robocasa
from robocasa.scripts.browse_mjcf_model import read_model
import robocasa.macros as macros
from robocasa.models.fixtures.fixture import FixtureType
from robosuite.wrappers import VisualizationWrapper

from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.scripts.collect_demos import collect_human_trajectory
from robocasa.models.scenes.scene_registry import (
    get_layout_path,
    get_style_path,
    LAYOUT_GROUPS_TO_IDS,
)
import re

NUM_DONE = 0


def get_all_style_configs():
    style_config_list = []
    for i in range(1, 61):
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


def get_valid_layout_ids(all_layout_ids=-3, filter_by=None):
    layout_id_list = []
    if isinstance(all_layout_ids, int) and all_layout_ids in LAYOUT_GROUPS_TO_IDS:
        # convert to list of layout ids
        all_layout_ids = LAYOUT_GROUPS_TO_IDS[all_layout_ids]
    for i in all_layout_ids:
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
        dict(env_name="OpenMicrowave"),
        dict(env_name="CloseMicrowave"),
        dict(env_name="TurnOnMicrowave"),
        dict(env_name="TurnOffMicrowave"),
    ],
    stove=[
        dict(env_name="PnPCounterToStove"),
        dict(env_name="PnPStoveToCounter"),
        dict(env_name="TurnOnStove"),
        dict(env_name="TurnOffStove"),
    ],
    stove_wide=[
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
        dict(env_name="StartCoffeeMachine"),
    ],
    dishwasher=[
        dict(env_name="OpenDishwasher"),
        dict(env_name="CloseDishwasher"),
        dict(env_name="SlideDishwasherRack"),
    ],
    oven=[
        dict(env_name="OpenOven"),
        dict(env_name="CloseOven"),
        dict(env_name="PnPCounterToOven"),
    ],
    fridge_french_door=[
        dict(env_name="OpenFridge"),
        dict(env_name="CloseFridge"),
    ],
    fridge_side_by_side=[
        dict(env_name="OpenFridge"),
        dict(env_name="CloseFridge"),
    ],
    fridge_bottom_freezer=[
        dict(env_name="OpenFridge"),
        dict(env_name="CloseFridge"),
    ],
    toaster=[
        dict(env_name="StartToaster"),
        dict(env_name="PnPToasterToCounter"),
    ],
    toaster_oven=[
        dict(env_name="AdjustToasterOvenTemperature"),
        dict(env_name="OpenToasterOvenDoor"),
        dict(env_name="CloseToasterOvenDoor"),
        dict(env_name="SlideToasterOvenRack"),
        dict(env_name="TurnOnToasterOven"),
        dict(env_name="PnPCounterToToasterOven"),
        dict(env_name="PnPToasterOvenToCounter"),
    ],
    stand_mixer=[
        dict(env_name="OpenStandMixerHead"),
        dict(env_name="CloseStandMixerHead"),
        dict(env_name="PnPCounterToStandMixer"),
    ],
    electric_kettle=[
        dict(env_name="CloseElectricKettleLid"),
        dict(env_name="OpenElectricKettleLid"),
        dict(env_name="TurnOnElectricKettle"),
    ],
    blender=[
        dict(
            env_name="Kitchen",
            init_robot_base_ref=FixtureType.BLENDER,
            enable_fixtures=["blender"],
        ),
    ],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    fixture_type_list = list(FIXTURE_TO_TEST_ENVS.keys())
    for fixture_type in fixture_type_list:
        parser.add_argument(f"--{fixture_type}", type=str, nargs="+")

    parser.add_argument("--layout", type=int, nargs="+", default=-3)
    parser.add_argument("--num_rollouts", type=int, default=3)
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

    now = datetime.now()
    current_date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    error_dict = {}

    num_total_tests = 0
    all_fixtures_dict = dict()
    for fixture_type in fixture_type_list:
        fixture_list = eval(f"args.{fixture_type}")

        if fixture_list is None:
            continue

        if "all" in fixture_list:
            yaml_path = os.path.join(
                robocasa.__path__[0],
                "models/assets/fixtures/fixture_registry",
                f"{fixture_type}.yaml",
            )
            with open(yaml_path) as yaml_f:
                fixture_registry = yaml.safe_load(yaml_f)
            fixture_list = list(fixture_registry.keys())
            fixture_list.remove("default")  # remove the "default fixture"

        all_fixtures_dict[fixture_type] = fixture_list
        env_kwargs_list = FIXTURE_TO_TEST_ENVS[fixture_type]
        num_total_tests += len(fixture_list) * len(env_kwargs_list)

    num_tests_completed = 0
    num_failures = 0
    base_log_dir = f"/tmp/robocasa_test_fixtures/{current_date_time}"
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    if args.device is None:
        # check if spacemouse is available
        spacemouse_found = False
        for device in hid.enumerate():
            vendor_id, product_id = device["vendor_id"], device["product_id"]
            if (
                vendor_id == macros.SPACEMOUSE_VENDOR_ID
                and product_id == macros.SPACEMOUSE_PRODUCT_ID
            ):
                spacemouse_found = True
                break
        args.device = "spacemouse" if spacemouse_found else "keyboard"

    for fixture_type in all_fixtures_dict.keys():
        error_dict[fixture_type] = dict()
        env_kwargs_list = FIXTURE_TO_TEST_ENVS[fixture_type]
        # get valid layouts for this fixture type
        layout_ids = get_valid_layout_ids(
            all_layout_ids=args.layout, filter_by=("type", fixture_type)
        )
        fixture_list = all_fixtures_dict[fixture_type]
        device = None
        for i, fixture_name in enumerate(fixture_list):
            if i < NUM_DONE:
                num_tests_completed += 1
                continue
            error_dict[fixture_type][fixture_name] = dict()
            style_configs = get_all_style_configs()
            for cfg in style_configs:
                cfg[fixture_type] = fixture_name
                if fixture_type == "blender":
                    # since the blender fixture has a correspnding lid that goes with it
                    # we need to update the style to include that too, this regex helps with that
                    # BlenderLid001 -> BlenderLid001 BlenderLid002 -> BlenderLid002, etc
                    cfg["blender_lid"] = re.sub(r"(\D+)(\d+)", r"\1Lid\2", fixture_name)

            fxtr_log_folder = f"/tmp/robocasa_test_fixtures/{current_date_time}/{fixture_type}/{fixture_name}"
            if not os.path.exists(fxtr_log_folder):
                os.makedirs(fxtr_log_folder)

            for env_i, env_kwargs in enumerate(env_kwargs_list):
                env_name = env_kwargs["env_name"]
                print(
                    colored(
                        f"[{num_tests_completed + 1}/{num_total_tests}] Type={fixture_type}, Model={fixture_name}, Env={env_name}",
                        color="yellow",
                    )
                )

                env = None

                try:
                    env = create_env(
                        render_onscreen=args.interactive,
                        seed=0,  # set seed=None to run unseeded
                        style_ids=style_configs,
                        layout_ids=layout_ids,
                        translucent_robot=True,
                        **env_kwargs,
                    )

                    # Wrap this with visualization wrapper
                    env = VisualizationWrapper(env)

                    if args.interactive is False:
                        info = run_random_rollouts(
                            env,
                            num_rollouts=args.num_rollouts,
                            num_steps=50,
                            video_path=f"{fxtr_log_folder}/{env_name}.mp4",
                        )
                    else:
                        # set up devices for interactive mode
                        # initialize device
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

                        # set up (updated) env for device
                        device.env = env

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
                except Exception:
                    error = traceback.format_exc()
                    error_message = f"Test {fixture_type}/{fixture_name}/{env_name} Failed:\n{error}\n"
                    error_dict[fixture_type][fixture_name][env_name] = error
                    print(colored(error_message, "red"))
                    with open(f"{fxtr_log_folder}/{env_name}_error.txt", "w") as f:
                        f.write(error)
                    with open(
                        f"{base_log_dir}/error_summary.txt", "a"
                    ) as log_summary_file:
                        log_summary_file.write(error_message)
                    num_failures += 1

                if env is not None:
                    env.close()
                    print()

                num_tests_completed += 1

    print("\n\n")
    if num_failures == 0:
        print(colored(f"{num_failures} exceptions encountered.", "green"))
    elif num_failures > 0:
        print(colored(f"{num_failures} exceptions encountered:\n", "red"))
        for fixture_type in error_dict:
            for fixture_name in error_dict[fixture_type]:
                for env_name in error_dict[fixture_type][fixture_name]:
                    error = error_dict[fixture_type][fixture_name][env_name]
                    error_message = f"Test {fixture_type}/{fixture_name}/{env_name} Failed:\n{error}\n"
                    print(colored(error_message, "red"))
