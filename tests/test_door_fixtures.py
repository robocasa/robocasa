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
import random

random.seed(0)

# exclude from testing
SKIP_FXTRS = ["handle_knob"]


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


def get_valid_layout_ids(all_layout_ids=-1, filter_by=None):
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


def get_types_from_cab_registry(cab_registry):
    handle_types, handle_texs, panel_types, panel_texs = [], [], [], []
    cab_registry.pop("default")
    for key, value in cab_registry.items():
        if "mix" in key:
            continue

        if key in SKIP_FXTRS:
            continue

        if "texture" in value:
            panel_texs.append(key)
        elif "panel_type" in value:
            panel_types.append(key)
        elif "handle_type" in value:
            handle_types.append(key)
        elif "handle_config" in value and "texture" in value["handle_config"]:
            handle_texs.append(key)
        else:
            raise ValueError("Unknown config type in cab")

    return {
        "cabinet_handle": handle_types,
        "handle_tex": handle_texs,
        "cabinet_panel": panel_types,
        "panel_tex": panel_texs,
    }


FIXTURE_TO_TEST_ENVS = dict(
    cabinet_panel=[
        dict(
            env_name="OpenCabinet",
        ),
        dict(
            env_name="CloseCabinet",
        ),
        dict(
            env_name="OpenDrawer",
        ),
        dict(
            env_name="CloseDrawer",
        ),
    ],
    cabinet_handle=[
        dict(
            env_name="OpenCabinet",
        ),
        dict(
            env_name="OpenDrawer",
        ),
    ],
)


def update_style_cabinets(
    panel_types, handle_types, style, panel_override, handle_override
):
    cab_config = style["cabinet"]

    KEEP_ARGS = ["default", "shelves"]
    remove_args = []

    # pop all override keys in the cab config. Want to only test one type of panel/handle
    for key in cab_config.keys():
        if key in KEEP_ARGS:
            continue
        remove_args.append(key)

    for key in remove_args:
        cab_config.pop(key)

    if isinstance(cab_config["default"], str):
        cab_config["default"] = [cab_config["default"]]
    panel_override = (
        random.choice(panel_types) if panel_override == "random" else panel_override
    )
    handle_override = (
        random.choice(handle_types) if handle_override == "random" else handle_override
    )

    removes = []

    for i in range(len(cab_config["default"])):
        if cab_config["default"][i] in panel_types:
            removes.append(i)
        elif cab_config["default"][i] in handle_types:
            removes.append(i)

    for remove_idx in reversed(removes):
        del cab_config["default"][remove_idx]
    cab_config["default"].append(panel_override)
    cab_config["default"].append(handle_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    fixture_type_list = list(FIXTURE_TO_TEST_ENVS.keys())
    for fixture_type in fixture_type_list:
        parser.add_argument(f"--{fixture_type}", type=str, nargs="+")

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
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
        yaml_path = os.path.join(
            robocasa.__path__[0],
            "models/assets/fixtures/fixture_registry",
            "cabinet.yaml",
        )
        with open(yaml_path) as yaml_f:
            fixture_registry = yaml.safe_load(yaml_f)
        cab_reg_types = get_types_from_cab_registry(fixture_registry)

        if "all" in fixture_list:
            fixture_list = cab_reg_types[fixture_type]

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

        fixture_list = all_fixtures_dict[fixture_type]
        device = None
        for fixture_name in fixture_list:
            error_dict[fixture_type][fixture_name] = dict()
            style_configs = get_all_style_configs()
            panel_override = (
                fixture_name if fixture_type == "cabinet_panel" else "random"
            )
            handle_override = (
                fixture_name if fixture_type == "cabinet_handle" else "random"
            )
            print(
                "Using cabinet panel ",
                panel_override,
                " and cabinet handle ",
                handle_override,
            )
            for cfg in style_configs:
                update_style_cabinets(
                    panel_types=cab_reg_types["cabinet_panel"],
                    handle_types=cab_reg_types["cabinet_handle"],
                    style=cfg,
                    panel_override=panel_override,
                    handle_override=handle_override,
                )

            fxtr_log_folder = f"/tmp/robocasa_test_fixtures/{current_date_time}/{fixture_type}/{fixture_name}"
            if not os.path.exists(fxtr_log_folder):
                os.makedirs(fxtr_log_folder)

            for env_i, env_kwargs in enumerate(env_kwargs_list):
                seed = random.randint(1, 1000)
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
                        seed=seed,
                        style_ids=style_configs,
                        translucent_robot=True,
                        layout_ids=args.layout,
                        **env_kwargs,
                    )

                    # Wrap this with visualization wrapper
                    env = VisualizationWrapper(env)

                    if args.interactive is False:
                        info = run_random_rollouts(
                            env,
                            num_rollouts=1,
                            num_steps=5,
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
