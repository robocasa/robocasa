import argparse
import json
import os
import traceback

import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from tqdm import tqdm


# TODO update args config on every iteration
def try_env(config, resets, actions):
    res = "success"
    env = None
    try:
        env = suite.make(**config)
        for _ in range(resets):
            env.reset()
            zero_action = np.array([0, 0, 0, 0, 0, 0, -1, -1])
            for _ in range(actions):
                obs, _, _, _ = env.step(zero_action)

        # also try setting model
        xml = env.sim.model.get_xml()
        env.reset_from_xml_string(xml)

    except Exception as e:
        res = traceback.format_exc()
        print(res)
    finally:
        # sometimes fail occurs on a make call
        if env:
            env.close()
    return res


def test_all_objects(config, asset_dir, resets, actions):
    log = {}
    object_cats = list(map(lambda x: os.path.join(asset_dir, x), os.listdir(asset_dir)))
    object_cats = list(filter(os.path.isdir, object_cats))
    models = []
    for cat in object_cats:
        cat_models = list(map(lambda x: os.path.join(cat, x), os.listdir(cat)))
        cat_models = list(
            filter(lambda x: os.path.exists(os.path.join(x, "model.xml")), cat_models)
        )
        models += cat_models

    for model in tqdm(models):
        config["obj_groups"] = os.path.join(model, "model.xml")
        obj = os.path.basename(model)
        res = try_env(config, resets, actions)
        if res != "success":
            print("failure for {obj}: {res}".format(obj=obj, res=res))
            log[obj] = res

    return log


def test_list_objects(config, asset_dir, objs, resets, actions):
    log = {}
    for obj in tqdm(objs):
        tokens = obj.split("_")
        obj_cat = "_".join(tokens[:-1])
        config["obj_group"] = os.path.join(asset_dir, obj_cat, obj, "bak_model.xml")
        res = try_env(config, resets, actions)
        if res != "success":
            log[obj] = res
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--asset_dir", type=str, required=True)
    parser.add_argument("--environment", type=str, default="KitchenPickPlace")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_group",
        type=str,
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument(
        "--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"]
    )

    parser.add_argument("--layout_id", type=int, default=0)
    parser.add_argument("--kitchen_style", default=0)
    parser.add_argument("--resets", default=1, type=int)
    parser.add_argument("--actions", default=1, type=int)
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller=args.controller)

    env_name = args.environment
    assert "Kitchen" in env_name

    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    mirror_actions = True
    config["kitchen_style"] = args.kitchen_style
    config["layout_id"] = args.layout_id
    if args.camera is None:
        args.camera = "robot0_frontview"

    config["has_renderer"] = True  # (args.renderer != "mjviewer")
    config["has_offscreen_renderer"] = True
    config["render_camera"] = args.camera
    #    config["renderer"] = args.renderer

    config["ignore_done"] = True
    config["use_camera_obs"] = True
    config["control_freq"] = 20
    log = test_all_objects(config, args.asset_dir, args.resets, args.actions)
    # with open("log3.json", 'r') as f:
    #     info = json.load(f)
    # log = test_list_objects(config, args.asset_dir, info.keys(), args.resets, args.actions)

    with open("log.json", "w") as f:
        json.dump(log, f, indent=4)
