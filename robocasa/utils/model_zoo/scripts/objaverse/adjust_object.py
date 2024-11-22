import argparse
from pynput.keyboard import Listener
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import numpy as np
import shutil
import json
import os

from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robocasa.utils.model_zoo.object_play_env import ObjectPlayEnv
from robosuite.devices import SpaceMouse
from update_bb_sites import update_bb_sites


with open("assets/default_object_scales.json", "r") as f:
    DEFAULT_OBJ_SCALES = json.load(f)

# global variable to indicate if program is running
running = True


class ObjectAdjustor:
    def __init__(
        self,
        xml_path,
        load_from_bak=False,
        scale_increment=0.05,
        rot_increment=45,
        instant_reload=True,
    ):
        # object loading
        self.xml_path = xml_path
        self.load_from_bak = load_from_bak
        self.first = True

        # env reset
        self.reset_now = False
        self.instant_reload = instant_reload

        # adjustments
        self.scale_increment = scale_increment
        self.scale_target = None
        self.scale_init = None

        self.rotations = {
            "@": [rot_increment, 0, 0],
            "2": [rot_increment / 2, 0, 0],
            "!": [-rot_increment, 0, 0],
            "1": [-rot_increment / 2, 0, 0],
            "$": [0, rot_increment, 0],
            "4": [0, rot_increment / 2, 0],
            "#": [0, -rot_increment, 0],
            "3": [0, -rot_increment / 2, 0],
            "^": [0, 0, rot_increment],
            "6": [0, 0, rot_increment / 2],
            "%": [0, 0, -rot_increment],
            "5": [0, 0, -rot_increment / 2],
        }
        self.rot_target = None
        self.rot_cum = np.array([0.0, 0.0, 0.0])
        self.rot_init = None

        self.display_controls()
        self.load_object()

    @staticmethod
    def display_controls():
        def print_command(char, info):
            print("{}\t{}".format(char.ljust(20), info))

        print("")
        print_command("Control", "Command")
        print_command("(-, +)", "(Increase, decrease) object scale slightly")
        print_command("Shift + (-, +)", "(Increase, decrease) object scale")
        # rotations
        print_command("(2, 4, 6)", "Rotate around (x, y, z)-axis clockwise slightly")
        print_command("Shift + (2, 4, 6)", "Rotate around (x, y, z) axis clockwise")
        print_command(
            "(1, 3, 5)", "Rotate around (x, y, z)-axis counter-clockwise slightly"
        )
        print_command(
            "Shift + (1, 3, 5)", "Rotate around (x, y, z) axis counter-clockwise"
        )
        # controls
        print_command("r", "Save transformations and reset")
        print_command("q", "Save transformations and quit")
        print("")

    def load_object(self):
        # extract initial scale and rotation
        self.scale_target, self.rot_init = self.get_initial_params()
        self.scale_init = self.scale_target

        # test if it's the first time that the object has been adjusted
        bak_path = os.path.join(os.path.dirname(self.xml_path), "bak_model.xml")
        if not os.path.exists(bak_path):
            # make backup copy if the original has not been saved
            shutil.copy(self.xml_path, bak_path)

            # use default scale if the object has not been seen before
            obj_name = os.path.basename(os.path.dirname(self.xml_path))
            obj_cat = obj_name.split("_")[0]
            if obj_cat in DEFAULT_OBJ_SCALES:
                self.scale_target = DEFAULT_OBJ_SCALES[obj_cat]
            else:
                self.scale_target = DEFAULT_OBJ_SCALES["other"]
            self.reset_now = True

        if self.first:
            print("\nLoaded object from:", self.xml_path)
            print(
                "Initial scale: {:.3f}\nInitial rotation: {}\n".format(
                    self.scale_target, self.rot_init
                )
            )
            self.first = False

    def get_initial_params(self):
        if self.load_from_bak:
            file_path = os.path.join(os.path.dirname(self.xml_path), "bak_model.xml")
        else:
            file_path = self.xml_path
        tree = ET.parse(file_path)
        asset = tree.getroot().find("asset")
        mesh = asset.findall("mesh")[0]

        scale = float(mesh.get("scale").split()[0])
        rot = mesh.get("refquat")
        if rot is None:
            rot = "1 0 0 0"
        rot = [float(x) for x in rot.split()]

        return scale, rot

    def on_press(self, key):
        # if currently reloading object, ignore key presses
        if self.reset_now:
            return

        try:
            prev_scale = self.scale_target
            rotation = None

            # scale adjustments
            if key.char == "-":
                self.scale_target -= self.scale_increment / 2
            elif key.char == "=":
                self.scale_target += self.scale_increment / 2
            elif key.char == "_":
                self.scale_target -= self.scale_increment
            elif key.char == "+":
                self.scale_target += self.scale_increment

            # rotation adjustments
            elif key.char in self.rotations:
                rotation = self.rotations[key.char]

            # reset
            elif key.char.lower() == "r":
                self.reset_now = True
                return
            elif key.char.lower() == "q":
                # update sites
                update_bb_sites(os.path.dirname(self.xml_path))

                self.reset_now = True
                global running
                running = False
                return

            if self.scale_target < 0:
                self.scale_target = prev_scale
                print("Cannot reduce scale further, current scale:", self.scale_target)
            elif self.scale_target != prev_scale:
                print("Current scale: {:.3f}".format(self.scale_target))

                if self.instant_reload:
                    self.reset_now = True

            if rotation is not None:
                self.rot_cum += rotation
                print("Cumulative rotations ([x, y, z]):", self.rot_cum)
                rotation = Rotation.from_euler("xyz", rotation, degrees=True)
                # [x, y, z, w] order seems to be reversed in robosuite
                rotation = rotation.as_quat()[::-1]

                if self.instant_reload:
                    self.reset_now = True
                    self.rot_target = rotation
                else:
                    if self.rot_target is not None:
                        self.rot_target = self.quat_multiply(rotation, self.rot_target)
                    else:
                        self.rot_target = rotation

        except AttributeError:
            pass

    def write_transformation(self, scale=None, rot=None):
        tree = ET.parse(self.xml_path)
        asset = tree.getroot().find("asset")
        meshes = asset.findall("mesh")

        # set scale and rotation
        for mesh in meshes:
            if scale is not None:
                mesh.set("scale", "{} {} {}".format(scale, scale, scale))
            if rot is not None:
                mesh.set("refquat", rot)

        # save new xml file
        tree.write(self.xml_path)
        print("Transformations saved\n")

        # save information in meta.json
        meta_path = os.path.join(os.path.dirname(self.xml_path), "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["scale"] = scale
        if rot is None:
            rot = "1 0 0 0"
        rot = [float(x) for x in rot.split()]
        meta["rot_quat"] = rot
        meta["rot"] = Rotation.from_quat(rot).as_matrix().tolist()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    @staticmethod
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return [w, x, y, z]

    def get_quat(self):
        return env.sim.data.get_body_xquat(
            env.model.mujoco_objects[0].name + "_main"
        ).copy()

    def reset(self):
        # check if changes have been made
        if self.scale_target != self.scale_init or self.rot_target is not None:
            # rotation is based on the original axis of the object (not current)
            if self.rot_target is not None:
                rot = self.quat_multiply(self.rot_target, self.rot_init)
            else:
                rot = self.rot_init
            rot = " ".join([str(x) for x in rot])
            self.write_transformation(self.scale_target, rot)

        # reset
        self.load_object()
        self.reset_now = False
        self.rot_target = None

    def start_episode(self, env, device):
        env.reset()
        if device is not None:
            device.start_control()

        while True:
            if device is not None:
                action, grasp = input2action(device=device, robot=env.robots[0])
            else:
                # zero action
                action = [0, 0, 0, 0, 0, 0, -1]

            # save transformations and reload environment
            if action is None or self.reset_now:
                if action is None:
                    global running
                    running = False
                self.reset()
                break
            # Run environment step
            env.step(action)
            env.render()

        env.close()


def create_env(args):
    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    # Create argument configuration
    config = {
        "robots": "Panda",
        "controller_configs": controller_config,
        "obj_mjcf_path": args.mjcf_path,
    }

    # Create environment
    env = ObjectPlayEnv(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        x_range=(0.0, 0.0),
        y_range=(0.0, 0.0),
        rotation=(0.0, 0.0),
    )
    return env


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf_path", type=str, required=True, help="path to object MJCF"
    )
    parser.add_argument("--default_scale", type=float, default=None)
    parser.add_argument("--scale_increment", type=float, default=0.02)
    parser.add_argument("--rot_increment", type=float, default=45)
    parser.add_argument("--default_rot", type=str, default=None)
    parser.add_argument(
        "--load_from_bak",
        action="store_true",
        help="Load from original object orientation and scale",
    )
    parser.add_argument(
        "--instant_reload",
        action="store_true",
        help="Whether to reload each time a change occurs",
    )

    parser.add_argument(
        "--device", type=str, default="dummy", choices=["dummy", "spacemouse"]
    )
    parser.add_argument(
        "--camera", type=str, default="agentview", help="Which camera to use"
    )
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )

    args = parser.parse_args()

    adjustor = ObjectAdjustor(
        args.mjcf_path,
        scale_increment=args.scale_increment,
        rot_increment=args.rot_increment,
        load_from_bak=args.load_from_bak,
        instant_reload=args.instant_reload,
    )

    if args.default_scale is not None or args.default_rot is not None:
        adjustor.write_transformation(scale=args.default_scale, rot=args.default_rot)
    if args.load_from_bak:
        args.mjcf_path = os.path.join(os.path.dirname(args.mjcf_path), "bak_model.xml")

    # initialize environment and device
    env = create_env(args)
    if args.device == "spacemouse":
        device = SpaceMouse(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dummy":
        device = None

    # monitor keyboard input
    listener = Listener(on_press=adjustor.on_press)
    listener.start()

    while running:
        adjustor.start_episode(env, device)
