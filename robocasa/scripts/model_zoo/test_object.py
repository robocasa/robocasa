import argparse
import numpy as np
from tqdm import trange
from time import time
import shutil
import os

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.devices import SpaceMouse
from robosuite.wrappers import DataCollectionWrapper


"""
This scripts performs the following tests:
    1. Detects and counts object "jumps" that occur based on a given threshold
    2. Records the frame rate of simulation rendering when using a given object
"""


class ObjectTester:
    def __init__(self, xml_path, num_steps=20):
        # object loading
        self.xml_path = xml_path
        self.num_steps = num_steps

    def start_episode(self, env, device):
        env.reset()
        if device is not None:
            device.start_control()

        def get_object_pos():
            # for object play environment - not fully implemented
            # print(len(env.model.mujoco_objects))
            # print([x.name for x in env.model.mujoco_objects])
            # object_name = env.model.mujoco_objects[0].name + "_main"
            # return env.sim.data.get_body_xpos(object_name).copy()

            # for kitchen environments
            obj_pos = env.sim.data.body_xpos[env.obj_body_id["obj"]]
            # only considering movements on the xy-plane
            obj_pos = obj_pos.copy()[:2]
            return obj_pos

        kitchen_env = "Kitchen" in str(env)
        zero_action = [0, 0, 0, 0, 0, 0, -1]
        if kitchen_env:
            zero_action.append(0)

        # advance simulation to see if object jumps
        start_pos = get_object_pos()
        for _ in range(self.num_steps):
            env.step(zero_action)
            # env.render()
        end_pos = get_object_pos()
        distance = np.sqrt(np.sum(np.square(end_pos - start_pos)))

        # advance simulation to test frame rate
        start = time()
        for _ in range(args.num_steps_fr):
            env.step(zero_action)
            env.render()
        elapsed = time() - start
        fps = args.num_steps_fr / elapsed

        # only used for debugging
        if device is not None:
            print("Distance moved:", distance)
            while True:
                action, grasp = input2action(
                    device=device,
                    robot=env.robots[0],
                    mirror_actions="Kitchen" in str(env),
                )
                if action is None:
                    break
                env.step(action)
                env.render()

        env.close()
        return distance, fps


def create_env(args, obj_config):
    start = time()
    env = suite.make(
        robots="Panda",
        obj_group=args.mjcf_path,
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        env_name=args.env,
        layout_id=args.layout_id,
        has_renderer=args.device is not None,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        # obj_default_config=obj_config
    )

    if args.use_wrapper:
        # use data collection wrapper, doesn't seem to make a difference
        env = DataCollectionWrapper(env, "/tmp/object_test_tmp")

    return env, time() - start


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=str,
        # the very "jumpy" carrot object
        default="/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/"
        "assets/shapenet_sem/carrots/78bd798a/model.xml",
        help="which object(s) to test, can be either path to a .xml or a category folder",
    )
    parser.add_argument(
        "--device", type=str, default="dummy", choices=["dummy", "spacemouse"]
    )
    parser.add_argument("--camera", type=str, default="robot0_agentview_center")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--layout_id", type=int, default=0)

    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--use_wrapper", action="store_true")

    parser.add_argument(
        "--num_steps_jump",
        type=int,
        default=20,
        help="how many steps to take for jump detection",
    )
    parser.add_argument(
        "--jump_threshold",
        type=float,
        default=0.03,
        help="how far the object has to move (on the x-y plane) for an episode to be counted as a jump",
    )
    parser.add_argument(
        "--num_steps_fr",
        type=int,
        default=100,
        help="how many steps to take to calculate average frame rate",
    )

    args = parser.parse_args()

    # object configurations for parameter search
    obj_configs = [{"density": [30]}]

    # allows searching through multiple environments
    envs = [
        ("KitchenPnPStoveTest", 0),
    ]
    if args.env is not None:
        envs = [(args.env, args.layout_id)]

    # initialize device
    if args.device == "spacemouse":
        device = SpaceMouse()
    else:
        device = None

    if os.path.isdir(args.mjcf):
        models = os.listdir(args.mjcf)
        models = [
            os.path.join(args.mjcf, model) for model in models if "." not in model
        ]
    else:
        models = [os.path.dirname(args.mjcf)]

    for model_folder in models:
        print("Testing mode:", model_folder)
        mjcf_path = os.path.join(model_folder, "model.xml")
        args.mjcf_path = mjcf_path

        adjustor = ObjectTester(args.mjcf_path, num_steps=args.num_steps_jump)

        for env_name, layout_id in envs:
            print("Using env: {}, layout id: {}".format(env_name, layout_id))
            args.env = env_name
            args.layout_id = layout_id

            for i in range(len(obj_configs)):
                config = obj_configs[i]
                distances = list()
                frame_rates = list()
                load_times = list()
                jumps = 0

                print("Object configuration:", config)
                t = trange(
                    args.num_episodes,
                    ncols=100,
                    desc="jumps detected: 0; fps: 0; load_time: 0",
                )

                for j in t:
                    env, load_time = create_env(args, obj_config=config)
                    distance, fps = adjustor.start_episode(env, device)

                    load_times.append(load_time)
                    distances.append(distance)
                    frame_rates.append(fps)
                    if distance > args.jump_threshold:
                        jumps += 1
                    t.set_description(
                        "jumps detected: {}; fps: {:.2f}; load_time: {:.2f}".format(
                            jumps, fps, load_time
                        )
                    )

                print(
                    "{} jumps detected, {:.2f}%".format(
                        jumps, jumps / args.num_episodes * 100
                    )
                )
                print(
                    "Average xy-plane distance: {:.4f}, threshold is {:.4f}".format(
                        np.mean(distances), args.jump_threshold
                    )
                )
                print("Average frame rate: {:.2f}".format(np.mean(frame_rates)))
                print("Average load time: {:.2f}".format(np.mean(load_times)))
            print()

    # delete generated data if using data collection wrapper
    if os.path.exists("/tmp/object_test_tmp"):
        shutil.rmtree("/tmp/object_test_tmp")
