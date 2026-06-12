"""GROOT / SONIC-compatible DDS bridge for RoboCasa G1Sonic.

The loop mirrors the GROOT MuJoCo simulator control boundary:

* subscribe rt/lowcmd for SONIC body motor commands
* subscribe rt/dex3/{left,right}/cmd for Dex3 hands
* compute PD torques from the DDS command and current MuJoCo joint state
* write torques directly to MuJoCo data.ctrl
* publish rt/lowstate, rt/secondary_imu, rt/odostate, and Dex3 hand states
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import time
from dataclasses import dataclass
from glob import glob
from typing import Any

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.robots.manipulators.g1_sonic_robot import XML_PATH
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

import robocasa  # noqa: F401
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallHotkeyHandler,
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)

from robocasa.integrations.sonic import dds_utils


BODY_JOINTS = (
    dds_utils.SIM_LEG_JOINTS
    + dds_utils.SIM_TORSO_JOINTS
    + dds_utils.SIM_LEFT_ARM_JOINTS
    + dds_utils.SIM_RIGHT_ARM_JOINTS
)
LEFT_HAND_JOINTS = dds_utils.SIM_LEFT_HAND_JOINTS
RIGHT_HAND_JOINTS = dds_utils.SIM_RIGHT_HAND_JOINTS
ALL_TORQUE_JOINTS = BODY_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

GROOT_BODY_STAND_Q = np.array(
    [
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

GROOT_MOTOR_KP = np.array(
    [
        150,
        150,
        150,
        200,
        40,
        40,
        150,
        150,
        150,
        200,
        40,
        40,
        250,
        250,
        250,
        100,
        100,
        40,
        40,
        20,
        20,
        20,
        100,
        100,
        40,
        40,
        20,
        20,
        20,
    ],
    dtype=float,
)

GROOT_MOTOR_KD = np.array(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        5,
        5,
        5,
        5,
        5,
        2,
        2,
        2,
        2,
        2,
        5,
        5,
        2,
        2,
        2,
        2,
        2,
    ],
    dtype=float,
)

GROOT_HAND_STAND_Q = np.zeros(7, dtype=float)
GROOT_HAND_KP = np.full(7, 1.5, dtype=float)
GROOT_HAND_KD = np.full(7, 0.2, dtype=float)


@dataclass
class TorqueMap:
    joint_to_actuator: dict[str, int]


def make_sonic_controller_config(args: argparse.Namespace):
    """Create a minimal controller config so robosuite can build the robot.

    The SONIC bridge bypasses these controllers and writes MuJoCo torques
    directly, but RoboCasa still expects a valid composite controller config
    while constructing the environment.
    """

    config = load_composite_controller_config(robot=args.robot)
    base_controller = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "input_type": "delta",
        "output_max": 0.05,
        "output_min": -0.05,
        "impedance_mode": "fixed",
        "kp_limits": [0, 10000],
        "damping_ratio_limits": [0, 10000],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "use_torque_compensation": False,
    }
    side_kp = [100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0] + [1.5] * 7
    side_kd = [5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0] + [0.2] * 7
    for side in ("right", "left"):
        part = dict(base_controller)
        part["kp"] = side_kp
        part["kd"] = side_kd
        part["gripper"] = {"type": "GRIP"}
        config["body_parts"][side] = part

    leg = dict(base_controller)
    leg["kp"] = GROOT_MOTOR_KP[:12].tolist()
    leg["kd"] = GROOT_MOTOR_KD[:12].tolist()
    config["body_parts"]["legs"] = leg

    torso = dict(base_controller)
    torso["kp"] = GROOT_MOTOR_KP[12:15].tolist()
    torso["kd"] = GROOT_MOTOR_KD[12:15].tolist()
    config["body_parts"]["torso"] = torso
    return config


def get_robocasa_camera_names(args: argparse.Namespace) -> list[str]:
    camera_names: list[str] = []
    record_videos = bool(
        getattr(args, "collect_demos", False)
        and getattr(args, "record_camera_videos", False)
    )
    if args.publish_head_camera or args.show_camera_views or record_videos:
        camera_names.append(args.head_camera_name)
    if args.publish_wrist_cameras or args.show_camera_views or record_videos:
        camera_names.extend([args.left_wrist_camera_name, args.right_wrist_camera_name])
    for camera_name in getattr(args, "video_camera_names", ()) or ():
        camera_names.append(camera_name)
    return list(dict.fromkeys(camera_names))


def make_env(args: argparse.Namespace):
    controller_config = make_sonic_controller_config(args)
    use_robosuite_viewer = args.render and args.viewer_backend == "robosuite"
    publish_images = (
        args.publish_head_camera or args.publish_wrist_cameras or args.show_camera_views
        or (args.collect_demos and args.record_camera_videos)
    )
    camera_names = get_robocasa_camera_names(args) or [args.head_camera_name]
    print(colored("Creating RoboCasa / robosuite environment with G1Sonic...", "yellow"))
    env = robosuite.make(
        env_name=args.task,
        robots=args.robot,
        controller_configs=controller_config,
        layout_ids=args.layout,
        style_ids=args.style,
        has_renderer=use_robosuite_viewer,
        has_offscreen_renderer=publish_images,
        render_camera=args.camera,
        render_collision_mesh=args.render_collision_mesh,
        render_visual_mesh=True,
        ignore_done=True,
        use_camera_obs=publish_images,
        camera_names=camera_names,
        camera_heights=args.head_camera_height,
        camera_widths=args.head_camera_width,
        camera_depths=False,
        control_freq=args.control_freq,
        renderer=args.renderer,
        gripper_types=None,
        translucent_robot=args.translucent_robot,
        seed=args.seed,
    )
    env = VisualizationWrapper(env)
    env = EnclosingWallRenderWrapper(env, alpha=args.enclosing_wall_alpha, enabled=False)
    return env


def hide_mujoco_debug_visuals(env) -> None:
    """Hide MuJoCo debug sites / frames from offscreen camera renders."""

    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return

    render_context = getattr(env.sim, "_render_context_offscreen", None)
    vopt = getattr(render_context, "vopt", None)
    if vopt is None:
        return

    if hasattr(vopt, "sitegroup"):
        vopt.sitegroup[:] = 0
    if hasattr(vopt, "frame"):
        vopt.frame = mujoco.mjtFrame.mjFRAME_NONE

    if hasattr(vopt, "flags"):
        for flag_name in (
            "mjVIS_CAMERA",
            "mjVIS_COM",
            "mjVIS_CONTACTFORCE",
            "mjVIS_CONTACTPOINT",
            "mjVIS_JOINT",
            "mjVIS_LIGHT",
            "mjVIS_PERTFORCE",
            "mjVIS_PERTOBJ",
        ):
            flag = getattr(mujoco.mjtVisFlag, flag_name, None)
            if flag is not None:
                vopt.flags[int(flag)] = 0


def get_robocasa_observations(env):
    hide_mujoco_debug_visuals(env)
    return env._get_observations(force_update=True)


def get_camera_rgb_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    image = obs.get(f"{camera_name}_image")
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        return None
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class SonicCollectionHotkeys:
    """Global hotkeys for SONIC demo collection.

    The bridge does not use robosuite's DataCollectionWrapper.step(), because the
    SONIC controller writes torques directly to MuJoCo. These hotkeys only mark
    episode boundaries for the direct-torque loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_requested = False
        self._stop_requested = False
        self._discard_requested = False
        self._listener = None

        try:
            from pynput.keyboard import Listener  # noqa: PLC0415
        except ImportError as exc:
            print(colored(f"WARNING: collection hotkeys disabled: {exc}", "yellow"))
            return

        def _on_release(key):
            char = getattr(key, "char", None)
            if char not in {"r", "q", "x"}:
                return
            with self._lock:
                if char == "r":
                    self._start_requested = True
                elif char == "q":
                    self._stop_requested = True
                elif char == "x":
                    self._discard_requested = True

        self._listener = Listener(on_release=_on_release)
        self._listener.start()
        print(
            colored(
                "SONIC collection hotkeys: r=start episode, q=finish episode, "
                "x=discard/reset episode, Ctrl+C=exit.",
                "yellow",
            )
        )

    def consume_start(self) -> bool:
        with self._lock:
            requested = self._start_requested
            self._start_requested = False
            return requested

    def consume_stop(self) -> bool:
        with self._lock:
            requested = self._stop_requested
            self._stop_requested = False
            return requested

    def consume_discard(self) -> bool:
        with self._lock:
            requested = self._discard_requested
            self._discard_requested = False
            return requested

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()


def _numeric_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype.kind in {"O", "U", "S"}:
        return None
    return arr


def _safe_video_name(camera_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in camera_name)


def _motor_command_arrays(msg: Any | None, count: int) -> dict[str, np.ndarray]:
    arrays = {
        "q": np.full(count, np.nan, dtype=np.float32),
        "dq": np.full(count, np.nan, dtype=np.float32),
        "tau": np.full(count, np.nan, dtype=np.float32),
        "kp": np.full(count, np.nan, dtype=np.float32),
        "kd": np.full(count, np.nan, dtype=np.float32),
        "mode": np.full(count, -1, dtype=np.int16),
    }
    if msg is None:
        return arrays

    motors = getattr(msg, "motor_cmd", ())
    for i in range(min(count, len(motors))):
        motor = motors[i]
        arrays["q"][i] = float(getattr(motor, "q", np.nan))
        arrays["dq"][i] = float(getattr(motor, "dq", np.nan))
        arrays["tau"][i] = float(getattr(motor, "tau", np.nan))
        arrays["kp"][i] = float(getattr(motor, "kp", np.nan))
        arrays["kd"][i] = float(getattr(motor, "kd", np.nan))
        arrays["mode"][i] = int(getattr(motor, "mode", -1))
    return arrays


def snapshot_sonic_commands(state: dds_utils.DdsCommandState) -> dict[str, Any]:
    now = time.monotonic()
    with state.lock:
        lowcmd = state.lowcmd
        left_hand_cmd = state.left_hand_cmd
        right_hand_cmd = state.right_hand_cmd
        lowcmd_age = now - state.lowcmd_time if state.lowcmd_time else np.nan
        left_hand_age = now - state.left_hand_time if state.left_hand_time else np.nan
        right_hand_age = now - state.right_hand_time if state.right_hand_time else np.nan

    snapshot: dict[str, Any] = {
        "command_wall_time": np.array(time.time(), dtype=np.float64),
        "lowcmd_valid": np.array(lowcmd is not None, dtype=np.bool_),
        "left_hand_cmd_valid": np.array(left_hand_cmd is not None, dtype=np.bool_),
        "right_hand_cmd_valid": np.array(right_hand_cmd is not None, dtype=np.bool_),
        "lowcmd_age": np.array(lowcmd_age, dtype=np.float32),
        "left_hand_cmd_age": np.array(left_hand_age, dtype=np.float32),
        "right_hand_cmd_age": np.array(right_hand_age, dtype=np.float32),
        "lowcmd_mode_pr": np.array(getattr(lowcmd, "mode_pr", -1), dtype=np.int16),
        "lowcmd_mode_machine": np.array(getattr(lowcmd, "mode_machine", -1), dtype=np.int16),
    }

    for prefix, msg, count in (
        ("lowcmd", lowcmd, 35),
        ("dex3_left_cmd", left_hand_cmd, len(LEFT_HAND_JOINTS)),
        ("dex3_right_cmd", right_hand_cmd, len(RIGHT_HAND_JOINTS)),
    ):
        for field, values in _motor_command_arrays(msg, count).items():
            snapshot[f"{prefix}_{field}"] = values
    return snapshot


class SonicRoboCasaDataCollector:
    """RoboCasa-style episode recorder for the SONIC direct-torque loop."""

    def __init__(self, args: argparse.Namespace, env_info: str):
        t_now = time.time()
        time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")
        time_str = f"{time_str}_{args.task}_sonic"

        self.demo_dir = os.path.join(args.directory, time_str)
        self.episodes_dir = os.path.join(self.demo_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

        self.args = args
        self.env_info = env_info
        self.recording = False
        self.ep_directory: str | None = None
        self.states: list[np.ndarray] = []
        self.action_infos: list[dict[str, Any]] = []
        self.successful = False
        self.success_hold_count = -1
        self.t = 0
        self.successful_episodes: list[str] = []
        self._video_paths: dict[str, str] = {}
        self._video_frame_count = 0
        self._missing_video_cameras: set[str] = set()

        with open(os.path.join(self.demo_dir, "env_info.json"), "w") as file:
            json.dump(json.loads(env_info), file, indent=2)

        print(colored(f"SONIC demo collection directory: {self.demo_dir}", "yellow"))

    @classmethod
    def for_existing_episode(
        cls,
        args: argparse.Namespace,
        env_info: str,
        ep_directory: str,
    ) -> "SonicRoboCasaDataCollector":
        collector = cls.__new__(cls)
        collector.args = args
        collector.env_info = env_info
        collector.ep_directory = ep_directory
        collector.episodes_dir = os.path.dirname(ep_directory)
        collector.demo_dir = os.path.dirname(collector.episodes_dir)
        collector.recording = False
        collector.states = []
        collector.action_infos = []
        collector.successful = False
        collector.success_hold_count = -1
        collector.t = 0
        collector.successful_episodes = []
        collector._video_paths = {}
        collector._video_frame_count = 0
        collector._missing_video_cameras = set()
        return collector

    def start_episode(self, env) -> None:
        if self.recording:
            print(colored("SONIC demo collection is already recording.", "yellow"))
            return

        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.episodes_dir, f"ep_{t1}_{t2}")
        os.makedirs(self.ep_directory)

        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang")
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))

        self.states = [np.array(env.sim.get_state().flatten())]
        self.action_infos = []
        self.successful = False
        self.success_hold_count = -1
        self.t = 0
        self._video_paths = {}
        self._video_frame_count = 0
        self._missing_video_cameras = set()
        self.recording = True
        print(colored(f"Started SONIC demo episode: {self.ep_directory}", "green"))

    def record_step(
        self,
        env,
        action: np.ndarray,
        state: dds_utils.DdsCommandState,
    ) -> bool:
        if not self.recording:
            return False

        self.t += 1
        video_frame_index = (
            self.t // self.args.video_skip
            if self.args.record_camera_videos and self.args.video_skip > 0
            else -1
        )

        if self.t % self.args.collect_freq == 0:
            self.states.append(np.array(env.sim.get_state().flatten()))
            info = snapshot_sonic_commands(state)
            info["actions"] = np.array(action, dtype=np.float32)
            info["sim_time"] = np.array(env.sim.data.time, dtype=np.float64)
            info["video_frame_index"] = np.array(video_frame_index, dtype=np.int32)
            self.action_infos.append(info)

        if env._check_success():
            self.successful = True
            if self.success_hold_count > 0:
                self.success_hold_count -= 1
            elif self.success_hold_count < 0:
                self.success_hold_count = self.args.success_hold_steps
        else:
            self.success_hold_count = -1

        if self.args.flush_freq > 0 and self.t % self.args.flush_freq == 0:
            self._flush()

        return bool(
            self.args.auto_stop_on_success
            and self.success_hold_count == 0
            and self.successful
        )

    def _flush(self) -> None:
        if self.ep_directory is None or not self.action_infos:
            return
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, f"state_{t1}_{t2}.npz")
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            env="G1SonicRoboCasa",
        )
        self.states = []
        self.action_infos = []

    def finish_episode(self, env, *, discard: bool = False) -> None:
        if not self.recording:
            print(colored("No active SONIC demo episode to finish.", "yellow"))
            return
        if self.ep_directory is None:
            return

        self.successful = bool(self.successful or env._check_success())
        self._flush()
        with open(os.path.join(self.ep_directory, "model.xml"), "w") as file:
            file.write(env.model.get_xml())
        with open(os.path.join(self.ep_directory, "ep_meta.json"), "w") as file:
            json.dump(env.get_ep_meta(), file)
        if not discard:
            self._render_episode_videos_offline(env)

        with open(os.path.join(self.ep_directory, "ep_stats.json"), "w") as file:
            json.dump({"success": self.successful, "discard": discard}, file)
        with open(os.path.join(self.ep_directory, "env_info.json"), "w") as file:
            json.dump(json.loads(self.env_info), file, indent=2)
        if self._video_paths:
            with open(os.path.join(self.ep_directory, "videos.json"), "w") as file:
                json.dump(
                    {
                        "fps": self.args.video_fps,
                        "skip": self.args.video_skip,
                        "height": self.args.video_height,
                        "width": self.args.video_width,
                        "frames": self._video_frame_count,
                        "paths": self._video_paths,
                    },
                    file,
                    indent=2,
                )

        ep_name = os.path.basename(self.ep_directory)
        include_episode = not discard and (
            self.successful or not self.args.collect_success_only
        )
        if include_episode:
            self.successful_episodes.append(ep_name)

        self.recording = False
        print(
            colored(
                f"Finished SONIC demo episode: {ep_name} "
                f"success={self.successful} included={include_episode}",
                "green" if include_episode else "yellow",
            )
        )
        self.gather_hdf5(verbose=True)

    def gather_hdf5(self, *, verbose: bool = False) -> str | None:
        return gather_sonic_demonstrations_as_hdf5(
            self.episodes_dir,
            self.demo_dir,
            self.env_info,
            successful_episodes=self.successful_episodes,
            verbose=verbose,
            out_name=self.args.output_hdf5_name,
        )

    def close(self, env) -> None:
        if self.recording:
            self.finish_episode(env)

    def _video_camera_names(self) -> list[str]:
        camera_names = self.args.video_camera_names
        if camera_names:
            return list(dict.fromkeys(camera_names))
        return [
            self.args.head_camera_name,
            self.args.left_wrist_camera_name,
            self.args.right_wrist_camera_name,
        ]

    def _load_episode_states(self) -> list[np.ndarray]:
        if self.ep_directory is None:
            return []
        states: list[np.ndarray] = []
        for state_file in sorted(glob(os.path.join(self.ep_directory, "state_*.npz"))):
            dic = np.load(state_file, allow_pickle=True)
            states.extend(np.asarray(dic["states"]))
        return states

    def _render_episode_videos_offline(self, env) -> None:
        self._video_paths = {}
        self._video_frame_count = 0
        self._missing_video_cameras = set()

        if not self.args.record_camera_videos or self.ep_directory is None:
            return

        try:
            import imageio  # noqa: PLC0415
        except ImportError as exc:
            print(colored(f"WARNING: camera video recording disabled: {exc}", "yellow"))
            return

        states = self._load_episode_states()
        if not states:
            print(colored("No states available for SONIC demo video rendering.", "yellow"))
            return

        videos_dir = os.path.join(self.ep_directory, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        available_camera_names = set(env.sim.model.camera_names)
        camera_names = []
        for camera_name in self._video_camera_names():
            if camera_name in available_camera_names:
                camera_names.append(camera_name)
            else:
                print(
                    colored(
                        f"Camera '{camera_name}' is not available for demo video recording. "
                        f"Available cameras: {env.sim.model.camera_names}",
                        "yellow",
                    )
                )
                self._missing_video_cameras.add(camera_name)

        if not camera_names:
            print(colored("No available cameras for SONIC demo video recording.", "yellow"))
            return

        mosaic_path = os.path.join(self.ep_directory, "video.mp4")
        video_writer = imageio.get_writer(mosaic_path, fps=self.args.video_fps)
        self._video_paths["mosaic"] = os.path.relpath(mosaic_path, self.ep_directory)

        camera_video_writers = {}
        for camera_name in camera_names:
            camera_path = os.path.join(videos_dir, f"{_safe_video_name(camera_name)}.mp4")
            camera_video_writers[camera_name] = imageio.get_writer(
                camera_path,
                fps=self.args.video_fps,
            )
            self._video_paths[camera_name] = os.path.relpath(camera_path, self.ep_directory)

        saved_robot_alpha = self._force_robot_opaque_for_video(env)
        live_state = np.array(env.sim.get_state().flatten())
        live_ctrl = np.array(env.sim.data.ctrl, dtype=float).copy()
        try:
            for state_index, sim_state in enumerate(states):
                if (
                    self.args.video_skip > 0
                    and state_index % self.args.video_skip != 0
                    and state_index != len(states) - 1
                ):
                    continue
                env.sim.set_state_from_flattened(sim_state)
                env.sim.forward()
                obs = get_robocasa_observations(env)
                frames: list[np.ndarray] = []
                for camera_name, writer in camera_video_writers.items():
                    frame = self._render_camera_frame(env, camera_name, obs=obs)
                    if frame is None:
                        continue
                    writer.append_data(frame)
                    frames.append(frame)
                if frames:
                    video_writer.append_data(np.concatenate(frames, axis=1))
                    self._video_frame_count += 1
            print(
                colored(
                    f"Rendered SONIC demo videos: {self._video_frame_count} frames",
                    "yellow",
                )
            )
        finally:
            video_writer.close()
            for writer in camera_video_writers.values():
                writer.close()
            self._restore_robot_alpha(env, saved_robot_alpha)
            env.sim.set_state_from_flattened(live_state)
            env.sim.data.ctrl[:] = live_ctrl
            env.sim.forward()

    def _force_robot_opaque_for_video(self, env) -> dict[int, float]:
        saved_alpha: dict[int, float] = {}
        model = env.sim.model
        for robot in env.robots:
            for geom_name in robot.robot_model.visual_geoms:
                if geom_name not in model.geom_names:
                    continue
                geom_id = model.geom_name2id(geom_name)
                saved_alpha[int(geom_id)] = float(model.geom_rgba[geom_id, 3])
                model.geom_rgba[geom_id, 3] = 1.0
        return saved_alpha

    def _restore_robot_alpha(self, env, saved_alpha: dict[int, float]) -> None:
        model = env.sim.model
        for geom_id, alpha in saved_alpha.items():
            if geom_id < model.ngeom:
                model.geom_rgba[geom_id, 3] = alpha

    def _render_camera_frame(self, env, camera_name: str, obs: dict[str, Any] | None = None) -> np.ndarray | None:
        if camera_name not in env.sim.model.camera_names:
            if camera_name not in self._missing_video_cameras:
                print(
                    colored(
                        f"Camera '{camera_name}' is not available for demo video recording. "
                        f"Available cameras: {env.sim.model.camera_names}",
                        "yellow",
                    )
                )
                self._missing_video_cameras.add(camera_name)
            return None
        if obs is None:
            obs = get_robocasa_observations(env)
        frame = get_camera_rgb_from_obs(obs, camera_name)
        if frame is None:
            obs_name = f"{camera_name}_image"
            if obs_name not in self._missing_video_cameras:
                print(
                    colored(
                        f"Observation '{obs_name}' is not available for demo video recording. "
                        f"Available observation keys: {tuple(obs.keys())}",
                        "yellow",
                    )
                )
                self._missing_video_cameras.add(obs_name)
            return None
        return frame

def gather_sonic_demonstrations_as_hdf5(
    directory: str,
    out_dir: str,
    env_info: str,
    successful_episodes: list[str] | None = None,
    verbose: bool = False,
    out_name: str = "demo.hdf5",
) -> str | None:
    """Gather SONIC direct-torque episodes into a RoboCasa-style HDF5 file."""

    try:
        import h5py  # noqa: PLC0415
        import mujoco  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("h5py and mujoco are required to gather SONIC demos.") from exc

    hdf5_path = os.path.join(out_dir, out_name)
    if verbose:
        print(colored(f"Saving SONIC hdf5 to {hdf5_path}", "yellow"))

    with h5py.File(hdf5_path, "w") as h5_file:
        grp = h5_file.create_group("data")
        num_eps = 0
        env_name = None

        for ep_directory in sorted(os.listdir(directory)):
            if successful_episodes is not None and ep_directory not in successful_episodes:
                continue

            ep_path = os.path.join(directory, ep_directory)
            state_paths = sorted(glob(os.path.join(ep_path, "state_*.npz")))
            states = []
            actions = []
            commands: dict[str, list[np.ndarray]] = {}

            for state_file in state_paths:
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                for action_info in dic["action_infos"]:
                    actions.append(action_info["actions"])
                    for key, value in action_info.items():
                        if key in {"actions", "actions_abs"}:
                            continue
                        arr = _numeric_array_or_none(value)
                        if arr is None:
                            continue
                        commands.setdefault(key, []).append(arr)

            if len(states) == 0:
                continue

            del states[-1]
            if len(states) != len(actions):
                print(
                    colored(
                        f"Skipping {ep_directory}: states/actions mismatch "
                        f"({len(states)} vs {len(actions)}).",
                        "red",
                    )
                )
                continue

            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")

            xml_path = os.path.join(ep_path, "model.xml")
            with open(xml_path, "r") as file:
                ep_data_grp.attrs["model_file"] = file.read()

            ep_meta_path = os.path.join(ep_path, "ep_meta.json")
            if os.path.exists(ep_meta_path):
                with open(ep_meta_path, "r") as file:
                    ep_data_grp.attrs["ep_meta"] = file.read()

            videos_meta_path = os.path.join(ep_path, "videos.json")
            if os.path.exists(videos_meta_path):
                with open(videos_meta_path, "r") as file:
                    ep_data_grp.attrs["videos"] = file.read()

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))

            command_grp = ep_data_grp.create_group("commands")
            for key, values in sorted(commands.items()):
                if len(values) != len(actions):
                    continue
                try:
                    command_grp.create_dataset(key, data=np.array(values))
                except TypeError:
                    print(colored(f"Skipping non-HDF5 command field: {key}", "yellow"))

        if verbose:
            print(colored(f"{num_eps} SONIC demos gathered so far", "cyan"))

        if num_eps == 0:
            return None

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["robocasa_version"] = robocasa.__version__
        grp.attrs["robosuite_version"] = robosuite.__version__
        grp.attrs["mujoco_version"] = mujoco.__version__
        grp.attrs["env"] = env_name or "G1SonicRoboCasa"
        grp.attrs["env_info"] = env_info
        grp.attrs["control_backend"] = "sonic_direct_torque"

    return hdf5_path


class TeleimagerZmqMultiCameraPublisher:
    def __init__(self, args: argparse.Namespace):
        try:
            import cv2  # noqa: PLC0415
            import zmq  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Publishing simulated cameras requires opencv-python and pyzmq "
                "in the robocasa environment."
            ) from exc

        self.cv2 = cv2
        self.zmq = zmq
        self.height = args.head_camera_height
        self.width = args.head_camera_width
        self.fps = args.head_camera_fps
        self.jpeg_quality = args.head_camera_jpeg_quality
        self._next_publish_time = 0.0
        self._missing_cameras: set[str] = set()

        self.cameras = {
            "head_camera": {
                "enabled": args.publish_head_camera,
                "model_name": args.head_camera_name,
                "zmq_port": args.head_camera_zmq_port,
                "webrtc_port": args.head_camera_webrtc_port,
            },
            "left_wrist_camera": {
                "enabled": args.publish_wrist_cameras,
                "model_name": args.left_wrist_camera_name,
                "zmq_port": args.left_wrist_zmq_port,
                "webrtc_port": args.left_wrist_webrtc_port,
            },
            "right_wrist_camera": {
                "enabled": args.publish_wrist_cameras,
                "model_name": args.right_wrist_camera_name,
                "zmq_port": args.right_wrist_zmq_port,
                "webrtc_port": args.right_wrist_webrtc_port,
            },
        }

        self.cam_config = {}
        for public_name, camera in self.cameras.items():
            self.cam_config[public_name] = {
                "enable_zmq": camera["enabled"],
                "zmq_port": camera["zmq_port"],
                "enable_webrtc": False,
                "webrtc_port": camera["webrtc_port"],
                "webrtc_codec": None,
                "type": "robocasa",
                "image_shape": [self.height, self.width],
                "binocular": False,
                "fps": self.fps,
            }

        self.context = zmq.Context()
        self.pub_sockets = {}
        for public_name, camera in self.cameras.items():
            if not camera["enabled"]:
                continue
            socket = self.context.socket(zmq.PUB)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.bind(f"tcp://{args.image_server_host}:{camera['zmq_port']}")
            self.pub_sockets[public_name] = socket

        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.setsockopt(zmq.LINGER, 0)
        self.rep_socket.bind(f"tcp://{args.image_server_host}:{args.image_config_port}")
        self._stop = threading.Event()
        self._config_thread = threading.Thread(target=self._serve_config, daemon=True)
        self._config_thread.start()

        enabled = ", ".join(
            f"{name}={camera['model_name']}:{camera['zmq_port']}"
            for name, camera in self.cameras.items()
            if camera["enabled"]
        )
        print(
            colored(
                f"Publishing RoboCasa cameras for XR on config port {args.image_config_port}: "
                f"{enabled}",
                "yellow",
            )
        )

    def _serve_config(self):
        poller = self.zmq.Poller()
        poller.register(self.rep_socket, self.zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=200))
            if self.rep_socket in events:
                self.rep_socket.recv()
                self.rep_socket.send_json(self.cam_config)

    def publish(self, env):
        now = time.monotonic()
        if now < self._next_publish_time:
            return
        self._next_publish_time = now + (1.0 / self.fps)

        camera_names = env.sim.model.camera_names
        if not camera_names:
            if "all" not in self._missing_cameras:
                print(colored("No MuJoCo cameras are available; image publishing is disabled.", "yellow"))
                self._missing_cameras.add("all")
            return

        obs = get_robocasa_observations(env)
        for public_name, socket in self.pub_sockets.items():
            camera_name = self.cameras[public_name]["model_name"]
            if camera_name not in camera_names:
                if camera_name not in self._missing_cameras:
                    print(
                        colored(
                            f"Camera '{camera_name}' is not available; skipping {public_name}. "
                            f"Available cameras: {camera_names}",
                            "yellow",
                        )
                    )
                    self._missing_cameras.add(camera_name)
                continue

            rgb = get_camera_rgb_from_obs(obs, camera_name)
            if rgb is None:
                obs_name = f"{camera_name}_image"
                if obs_name not in self._missing_cameras:
                    print(
                        colored(
                            f"Observation '{obs_name}' is not available; skipping {public_name}. "
                            f"Available observation keys: {tuple(obs.keys())}",
                            "yellow",
                        )
                    )
                    self._missing_cameras.add(obs_name)
                continue
            bgr = rgb[:, :, ::-1]
            ok, encoded = self.cv2.imencode(
                ".jpg",
                bgr,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok:
                try:
                    socket.send(encoded.tobytes(), flags=self.zmq.NOBLOCK)
                except self.zmq.Again:
                    pass

    def close(self):
        self._stop.set()
        self._config_thread.join(timeout=1.0)
        for socket in self.pub_sockets.values():
            socket.close(0)
        self.rep_socket.close(0)
        self.context.term()


class SimCameraMosaicViewer:
    def __init__(self, env, args: argparse.Namespace):
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._image = Image
        self._image_tk = ImageTk
        self.height = args.camera_view_height
        self.width = args.camera_view_width
        self.fps = args.camera_view_fps
        self.overview_distance = args.overview_camera_distance
        self.overview_azimuth = args.overview_camera_azimuth
        self.overview_elevation = args.overview_camera_elevation
        self.overview_track_body_id = (
            self._find_overview_track_body(env) if args.show_overview_camera else None
        )
        self._next_render_time = 0.0
        self._closed = False
        camera_specs = []
        if args.show_overview_camera:
            camera_specs.append(("overview", None))
        camera_specs += [
            ("head", args.head_camera_name),
            ("left wrist", args.left_wrist_camera_name),
            ("right wrist", args.right_wrist_camera_name),
        ]

        camera_names = env.sim.model.camera_names
        self.cameras = []
        for label, camera_name in camera_specs:
            if camera_name is None:
                self.cameras.append((label, camera_name))
                continue
            if camera_name not in camera_names:
                print(
                    colored(
                        f"Camera '{camera_name}' is not available in MuJoCo; "
                        f"available cameras: {camera_names}",
                        "yellow",
                    )
                )
                continue

            self.cameras.append((label, camera_name))

        if not self.cameras:
            raise RuntimeError(
                "No requested robot cameras are available; cannot show camera views."
            )

        self.root = tk.Tk()
        self.root.title("RoboCasa G1Sonic views: " + " | ".join(label for label, _ in self.cameras))
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        self.image_labels = []
        num_columns = 2 if len(self.cameras) > 2 else max(1, len(self.cameras))
        for index, (label, _) in enumerate(self.cameras):
            frame = tk.Frame(self.root)
            frame.grid(row=index // num_columns, column=index % num_columns, padx=2, pady=2)
            tk.Label(frame, text=label).pack()
            image_label = tk.Label(frame)
            image_label.pack()
            self.image_labels.append(image_label)
        self.root.update_idletasks()
        self.root.update()

        print(
            colored(
                "Showing RoboCasa camera mosaic: "
                + ", ".join(
                    f"{label}={camera_name or 'tracking overview'}"
                    for label, camera_name in self.cameras
                ),
                "yellow",
            )
        )

    @staticmethod
    def _find_overview_track_body(env):
        for body_name in ("robot0_pelvis", "robot0_base", "robot0_torso_link"):
            if body_name in env.sim.model.body_names:
                return env.sim.model.body_name2id(body_name)
        return None

    def _render_overview(self, env):
        import mujoco

        hide_mujoco_debug_visuals(env)
        render_context = env.sim._render_context_offscreen
        cam = render_context.cam
        if self.overview_track_body_id is not None:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.overview_track_body_id
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = np.array([0.0, 0.0, 1.0])
        cam.distance = self.overview_distance
        cam.azimuth = self.overview_azimuth
        cam.elevation = self.overview_elevation
        return env.sim.render(
            height=self.height,
            width=self.width,
            camera_name=None,
        )[::-1]

    def _request_close(self):
        self._closed = True

    def sync(self, env) -> bool:
        now = time.monotonic()
        if now < self._next_render_time:
            return True
        self._next_render_time = now + (1.0 / self.fps)

        if self._closed:
            return False

        obs = None
        try:
            if any(camera_name is not None for _, camera_name in self.cameras):
                obs = get_robocasa_observations(env)
            for image_label, (_, camera_name) in zip(self.image_labels, self.cameras):
                if camera_name is None:
                    rgb = self._render_overview(env)
                else:
                    rgb = get_camera_rgb_from_obs(obs, camera_name)
                    if rgb is None:
                        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                image = self._image.fromarray(rgb)
                photo = self._image_tk.PhotoImage(image=image)
                image_label.configure(image=photo)
                image_label.image = photo
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            return False
        return True

    def close(self):
        self._closed = True
        if getattr(self, "root", None) is not None:
            try:
                self.root.destroy()
            except self._tk.TclError:
                pass
            self.root = None


class PassiveMujocoViewer:
    """Small wrapper around MuJoCo's passive viewer for direct sim.step loops."""

    def __init__(self, env) -> None:
        import mujoco
        from mujoco import viewer as mujoco_viewer

        self._mujoco = mujoco
        self._handle = mujoco_viewer.launch_passive(
            env.sim.model._model,
            env.sim.data._data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._handle.opt.geomgroup[0] = 1 if getattr(env, "render_collision_mesh", False) else 0
        self._handle.opt.geomgroup[1] = 1
        self._handle.opt.sitegroup[:] = 0
        self._handle.opt.frame = self._mujoco.mjtFrame.mjFRAME_NONE
        self._configure_camera(env)

    def _configure_camera(self, env) -> None:
        body_id = None
        for body_name in ("robot0_pelvis", "robot0_base", "robot0_torso_link"):
            if body_name in env.sim.model.body_names:
                body_id = env.sim.model.body_name2id(body_name)
                break
        if body_id is not None:
            self._handle.cam.type = self._mujoco.mjtCamera.mjCAMERA_TRACKING
            self._handle.cam.trackbodyid = body_id
        self._handle.cam.distance = 4.0
        self._handle.cam.azimuth = 135.0
        self._handle.cam.elevation = -20.0
        self._handle.sync()

    def sync(self) -> bool:
        if not self._handle.is_running():
            return False
        self._handle.sync()
        return True

    def close(self) -> None:
        self._handle.close()


def open_sim_viewer(env, args: argparse.Namespace):
    if args.viewer_backend == "robosuite":
        print(colored("Opening robosuite MuJoCo viewer...", "yellow"))
        env.render()
        print(colored("robosuite MuJoCo viewer render loop started.", "green"))
        return None

    print(colored("Opening MuJoCo passive viewer...", "yellow"))
    viewer = PassiveMujocoViewer(env)
    print(colored("MuJoCo passive viewer render loop started.", "green"))
    return viewer


def _joint_q(env, joint_name: str) -> float:
    return float(env.sim.data.get_joint_qpos(joint_name))


def _joint_dq(env, joint_name: str) -> float:
    return float(env.sim.data.get_joint_qvel(joint_name))


def _set_joint_state(env, joint_name: str, q: float, dq: float = 0.0) -> None:
    if joint_name not in env.sim.model.joint_names:
        return
    env.sim.data.set_joint_qpos(joint_name, float(q))
    env.sim.data.set_joint_qvel(joint_name, float(dq))


def reset_sonic_stand_state(env) -> None:
    for joint_name, q in zip(BODY_JOINTS, GROOT_BODY_STAND_Q):
        _set_joint_state(env, joint_name, q)
    for joint_name in LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS:
        _set_joint_state(env, joint_name, 0.0)
    env.sim.forward()


def reset_sonic_episode_env(env, args: argparse.Namespace) -> TorqueMap:
    env.reset()
    hide_mujoco_debug_visuals(env)
    env.sim.model.opt.timestep = 1.0 / args.sim_frequency
    reset_sonic_stand_state(env)
    return build_torque_map(env)


def simulation_is_unstable(env, threshold: float = 1e6) -> bool:
    for name, values in (
        ("qpos", env.sim.data.qpos),
        ("qvel", env.sim.data.qvel),
        ("qacc", env.sim.data.qacc),
    ):
        arr = np.asarray(values)
        if not np.all(np.isfinite(arr)):
            print(colored(f"MuJoCo state contains NaN/Inf in {name}; resetting robot stand state.", "red"))
            return True
        if arr.size and float(np.max(np.abs(arr))) > threshold:
            print(
                colored(
                    f"MuJoCo state has huge value in {name} "
                    f"(max={float(np.max(np.abs(arr))):.3e}); resetting robot stand state.",
                    "red",
                )
            )
            return True
    return False


def build_torque_map(env) -> TorqueMap:
    missing_joints = [name for name in ALL_TORQUE_JOINTS if name not in env.sim.model.joint_names]
    if missing_joints:
        raise RuntimeError(f"G1Sonic model is missing joints: {missing_joints}")

    joint_to_actuator: dict[str, int] = {}
    for joint_name in ALL_TORQUE_JOINTS:
        joint_id = env.sim.model.joint_name2id(joint_name)
        actuator_id = None
        for idx in range(env.sim.model.nu):
            trn = env.sim.model.actuator_trnid[idx]
            if int(trn[0]) == joint_id:
                actuator_id = idx
                break
        if actuator_id is None:
            raise RuntimeError(f"No actuator drives joint {joint_name}")
        joint_to_actuator[joint_name] = actuator_id
    return TorqueMap(joint_to_actuator=joint_to_actuator)


def _fresh_command(command_time: float, timeout: float) -> bool:
    return bool(command_time) and time.monotonic() - command_time <= timeout


def _clip_ctrl(env, actuator_id: int, torque: float) -> float:
    if not np.isfinite(torque):
        return 0.0
    low, high = env.sim.model.actuator_ctrlrange[actuator_id]
    if high > low:
        return float(np.clip(torque, low, high))
    return float(torque)


def _set_ctrl_for_joint(env, torque_map: TorqueMap, joint_name: str, torque: float) -> None:
    actuator_id = torque_map.joint_to_actuator[joint_name]
    env.sim.data.ctrl[actuator_id] = _clip_ctrl(env, actuator_id, torque)


def _pd_torque(q: float, dq: float, q_des: float, dq_des: float, tau_ff: float, kp: float, kd: float) -> float:
    values = (q, dq, q_des, dq_des, tau_ff, kp, kd)
    if not all(np.isfinite(v) for v in values):
        return 0.0
    return float(tau_ff + kp * (q_des - q) + kd * (dq_des - dq))


def compute_sonic_torques(
    env,
    state: dds_utils.DdsCommandState,
    args: argparse.Namespace,
) -> dict[str, float]:
    now = time.monotonic()
    with state.lock:
        lowcmd = state.lowcmd
        left_hand_cmd = state.left_hand_cmd
        right_hand_cmd = state.right_hand_cmd
        lowcmd_age = now - state.lowcmd_time if state.lowcmd_time else None
        left_hand_age = now - state.left_hand_time if state.left_hand_time else None
        right_hand_age = now - state.right_hand_time if state.right_hand_time else None

    torques: dict[str, float] = {}

    have_lowcmd = lowcmd is not None and (lowcmd_age is None or lowcmd_age <= args.command_timeout)
    for motor_id, joint_name in enumerate(BODY_JOINTS):
        q = _joint_q(env, joint_name)
        dq = _joint_dq(env, joint_name)
        if have_lowcmd:
            motor = lowcmd.motor_cmd[motor_id]
            q_des = float(motor.q)
            dq_des = float(motor.dq)
            tau_ff = float(motor.tau)
            kp = float(motor.kp)
            kd = float(motor.kd)
        elif args.hold_default_stand:
            q_des = float(GROOT_BODY_STAND_Q[motor_id])
            dq_des = 0.0
            tau_ff = 0.0
            kp = float(GROOT_MOTOR_KP[motor_id])
            kd = float(GROOT_MOTOR_KD[motor_id])
        else:
            continue
        torques[joint_name] = _pd_torque(q, dq, q_des, dq_des, tau_ff, kp, kd)

    have_left_hand = (
        left_hand_cmd is not None
        and (left_hand_age is None or left_hand_age <= args.command_timeout)
    )
    have_right_hand = (
        right_hand_cmd is not None
        and (right_hand_age is None or right_hand_age <= args.command_timeout)
    )

    torques.update(
        _compute_hand_torques(
            env,
            LEFT_HAND_JOINTS,
            left_hand_cmd,
            have_left_hand,
            args,
        )
    )
    torques.update(
        _compute_hand_torques(
            env,
            RIGHT_HAND_JOINTS,
            right_hand_cmd,
            have_right_hand,
            args,
        )
    )
    return torques


def apply_sonic_torques(
    env,
    torque_map: TorqueMap,
    state: dds_utils.DdsCommandState,
    args: argparse.Namespace,
) -> np.ndarray:
    env.sim.data.ctrl[:] = 0.0
    for joint_name, torque in compute_sonic_torques(env, state, args).items():
        _set_ctrl_for_joint(env, torque_map, joint_name, torque)
    return np.array(env.sim.data.ctrl, dtype=np.float32).copy()


def _compute_hand_torques(
    env,
    joint_names: list[str],
    handcmd: Any | None,
    have_cmd: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    torques: dict[str, float] = {}
    for motor_id, joint_name in enumerate(joint_names):
        q = _joint_q(env, joint_name)
        dq = _joint_dq(env, joint_name)
        if have_cmd:
            motor = handcmd.motor_cmd[motor_id]
            q_des = float(motor.q) * args.hand_scale + args.hand_offset
            dq_des = float(motor.dq)
            tau_ff = float(motor.tau)
            kp = float(motor.kp)
            kd = float(motor.kd)
        elif args.hold_default_stand:
            q_des = float(GROOT_HAND_STAND_Q[motor_id])
            dq_des = 0.0
            tau_ff = 0.0
            kp = float(GROOT_HAND_KP[motor_id])
            kd = float(GROOT_HAND_KD[motor_id])
        else:
            continue
        torques[joint_name] = _pd_torque(q, dq, q_des, dq_des, tau_ff, kp, kd)
    return torques


def _apply_hand_torques(
    env,
    torque_map: TorqueMap,
    joint_names: list[str],
    handcmd: Any | None,
    have_cmd: bool,
    args: argparse.Namespace,
) -> None:
    for joint_name, torque in _compute_hand_torques(
        env,
        joint_names,
        handcmd,
        have_cmd,
        args,
    ).items():
        _set_ctrl_for_joint(env, torque_map, joint_name, torque)


class GrootStatePublisher:
    def __init__(self, args: argparse.Namespace):
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher  # noqa: PLC0415
            from unitree_sdk2py.idl.default import (  # noqa: PLC0415
                unitree_hg_msg_dds__HandState_,
                unitree_hg_msg_dds__IMUState_,
                unitree_hg_msg_dds__LowState_,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
                HandState_,
                IMUState_,
                LowState_,
            )
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2py state message constructors are required for SONIC bridge."
            ) from exc

        try:
            from unitree_sdk2py.idl.default import (  # noqa: PLC0415
                unitree_hg_msg_dds__OdoState_,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import OdoState_  # noqa: PLC0415
        except ImportError:
            unitree_hg_msg_dds__OdoState_ = None
            OdoState_ = None

        self.lowstate_ctor = unitree_hg_msg_dds__LowState_
        self.handstate_ctor = unitree_hg_msg_dds__HandState_
        self.imu_ctor = unitree_hg_msg_dds__IMUState_
        self.odo_ctor = unitree_hg_msg_dds__OdoState_
        self.mode_machine = args.mode_machine
        self.publish_log_interval = args.publish_log_interval
        self._publish_count = 0
        self._last_publish_log_time = 0.0

        self.lowstate_pub = ChannelPublisher(args.lowstate_topic, LowState_)
        self.lowstate_pub.Init()
        self.secondary_imu_pub = ChannelPublisher(args.secondary_imu_topic, IMUState_)
        self.secondary_imu_pub.Init()
        self.odostate_pub = None
        if OdoState_ is not None:
            self.odostate_pub = ChannelPublisher(args.odostate_topic, OdoState_)
            self.odostate_pub.Init()
        self.left_hand_pub = ChannelPublisher(args.left_hand_state_topic, HandState_)
        self.left_hand_pub.Init()
        self.right_hand_pub = ChannelPublisher(args.right_hand_state_topic, HandState_)
        self.right_hand_pub.Init()

        odostate_label = args.odostate_topic if self.odostate_pub is not None else "odostate skipped"
        print(
            colored(
                "Publishing GROOT-compatible simulated state on "
                f"{args.lowstate_topic}, {args.secondary_imu_topic}, "
                f"{odostate_label}, rt/dex3/*/state",
                "yellow",
            )
        )

    def publish(self, env) -> None:
        lowstate = self.lowstate_ctor()
        lowstate.mode_machine = self.mode_machine
        dds_utils.UnitreeStatePublisher._set_imu_state(lowstate, env)
        for motor_id, joint_name in enumerate(BODY_JOINTS):
            dds_utils.UnitreeStatePublisher._set_motor_state(lowstate, motor_id, env, joint_name)
        try:
            lowstate.tick = int(env.sim.data.time * 1e3)
        except AttributeError:
            pass
        self.lowstate_pub.Write(lowstate)

        left = self.handstate_ctor()
        right = self.handstate_ctor()
        dds_utils.UnitreeStatePublisher._set_hand_states(
            left,
            env,
            LEFT_HAND_JOINTS,
            side="left",
            hand_mapping_profile="raw",
            fallback_q=0.0,
        )
        dds_utils.UnitreeStatePublisher._set_hand_states(
            right,
            env,
            RIGHT_HAND_JOINTS,
            side="right",
            hand_mapping_profile="raw",
            fallback_q=0.0,
        )
        self.left_hand_pub.Write(left)
        self.right_hand_pub.Write(right)

        self._publish_secondary_imu(env)
        self._publish_odo(env)

        self._publish_count += 1
        now = time.monotonic()
        if (
            self.publish_log_interval > 0
            and now - self._last_publish_log_time >= self.publish_log_interval
        ):
            self._last_publish_log_time = now
            print(colored(f"Published GROOT lowstate packets: {self._publish_count}", "cyan"))

    def _publish_secondary_imu(self, env) -> None:
        msg = self.imu_ctor()
        body_name = _first_existing_body(env, ["robot0_torso_link", "torso_link"])
        if body_name is None:
            msg.quaternion = [1.0, 0.0, 0.0, 0.0]
            msg.gyroscope = [0.0, 0.0, 0.0]
        else:
            msg.quaternion[:] = np.asarray(env.sim.data.get_body_xquat(body_name), dtype=float)
            msg.gyroscope[:] = _body_angular_velocity_body(env, body_name)
        self.secondary_imu_pub.Write(msg)

    def _publish_odo(self, env) -> None:
        if self.odostate_pub is None or self.odo_ctor is None:
            return
        msg = self.odo_ctor()
        body_name = _first_existing_body(env, ["robot0_pelvis", "robot0_base", "robot0_torso_link"])
        if body_name is not None:
            msg.position[:] = np.asarray(env.sim.data.get_body_xpos(body_name), dtype=float)
            msg.orientation[:] = np.asarray(env.sim.data.get_body_xquat(body_name), dtype=float)
            try:
                msg.linear_velocity[:] = np.asarray(env.sim.data.get_body_xvelp(body_name), dtype=float)
                msg.angular_velocity[:] = np.asarray(env.sim.data.get_body_xvelr(body_name), dtype=float)
            except (AttributeError, ValueError):
                msg.linear_velocity[:] = [0.0, 0.0, 0.0]
                msg.angular_velocity[:] = [0.0, 0.0, 0.0]
        try:
            msg.tick = int(env.sim.data.time * 1e3)
        except AttributeError:
            pass
        self.odostate_pub.Write(msg)


def _first_existing_body(env, names: list[str]) -> str | None:
    body_names = getattr(env.sim.model, "body_names", ())
    for name in names:
        if name in body_names:
            return name
    return None


def _body_angular_velocity_body(env, body_name: str) -> np.ndarray:
    rot = np.array(env.sim.data.get_body_xmat(body_name), dtype=float).reshape(3, 3)
    try:
        ang_vel_world = np.array(env.sim.data.get_body_xvelr(body_name), dtype=float)
    except (AttributeError, ValueError):
        ang_vel_world = np.zeros(3, dtype=float)
    return rot.T @ ang_vel_world


def log_sonic_command_state(state: dds_utils.DdsCommandState, args: argparse.Namespace) -> None:
    now = time.monotonic()
    with state.lock:
        lowcmd_age = now - state.lowcmd_time if state.lowcmd_time else None
        left_age = now - state.left_hand_time if state.left_hand_time else None
        right_age = now - state.right_hand_time if state.right_hand_time else None
        lowcmd = state.lowcmd
    if lowcmd is None:
        print(colored(f"SONIC command log: waiting for LowCmd on {args.cmd_topic}", "yellow"))
        return
    q0 = float(lowcmd.motor_cmd[0].q)
    kp0 = float(lowcmd.motor_cmd[0].kp)
    print(
        colored(
            "SONIC command log: "
            f"lowcmd_age={lowcmd_age or 0.0:.3f}s "
            f"left_hand_age={left_age or 0.0:.3f}s "
            f"right_hand_age={right_age or 0.0:.3f}s "
            f"motor0_q={q0:.3f} motor0_kp={kp0:.1f}",
            "cyan",
        )
    )


def rerender_existing_episode(args: argparse.Namespace) -> None:
    ep_directory = os.path.abspath(args.rerender_episode)
    env_info_path = os.path.join(ep_directory, "env_info.json")
    if not os.path.isdir(ep_directory):
        raise FileNotFoundError(f"Episode directory not found: {ep_directory}")
    if not os.path.exists(env_info_path):
        raise FileNotFoundError(f"Episode env_info.json not found: {env_info_path}")

    with open(env_info_path, "r") as file:
        env_info_data = json.load(file)

    args.task = env_info_data.get("env_name", args.task)
    args.robot = env_info_data.get("robots", args.robot)
    args.layout = int(env_info_data.get("layout_ids", args.layout))
    args.style = int(env_info_data.get("style_ids", args.style))
    stored_args = env_info_data.get("args", {})
    if args.video_camera_names is None:
        args.video_camera_names = stored_args.get("video_camera_names")

    args.collect_demos = True
    args.record_camera_videos = True
    args.render = False
    args.show_camera_views = False
    args.publish_head_camera = False
    args.publish_wrist_cameras = False

    print(
        colored(
            "Re-rendering SONIC episode videos: "
            f"episode={ep_directory} task={args.task} robot={args.robot} "
            f"layout={args.layout} style={args.style}",
            "yellow",
        )
    )
    env = make_env(args)
    try:
        reset_sonic_episode_env(env, args)
        collector = SonicRoboCasaDataCollector.for_existing_episode(
            args,
            json.dumps(env_info_data),
            ep_directory,
        )
        collector._render_episode_videos_offline(env)
        if collector._video_paths:
            videos_meta = {
                "fps": args.video_fps,
                "skip": args.video_skip,
                "height": args.video_height,
                "width": args.video_width,
                "frames": collector._video_frame_count,
                "paths": collector._video_paths,
            }
            with open(os.path.join(ep_directory, "videos.json"), "w") as file:
                json.dump(videos_meta, file, indent=2)
            print(colored(f"Re-rendered video: {os.path.join(ep_directory, 'video.mp4')}", "green"))
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--robot", default="G1Sonic", choices=["G1Sonic"])
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--viewer-backend", choices=["passive", "robosuite"], default="passive")
    parser.add_argument("--render-collision-mesh", action="store_true")
    parser.add_argument("--translucent-robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enclosing-wall-alpha", type=float, default=0.15)
    parser.add_argument("--control-freq", type=float, default=50.0)
    parser.add_argument("--sim-frequency", type=float, default=200.0)
    parser.add_argument("--viewer-frequency", type=float, default=50.0)
    parser.add_argument("--collect-demos", action="store_true")
    parser.add_argument("--directory", type=str, default="./datasets")
    parser.add_argument("--output-hdf5-name", default="demo.hdf5")
    parser.add_argument("--collect-freq", type=int, default=1)
    parser.add_argument("--flush-freq", type=int, default=0)
    parser.add_argument("--collect-success-only", action="store_true")
    parser.add_argument("--auto-start-recording", action="store_true")
    parser.add_argument("--auto-stop-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--record-camera-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerender-episode", default=None)
    parser.add_argument("--video-camera-names", nargs="*", default=None)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-skip", type=int, default=10)

    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--cmd-topic", default="rt/lowcmd")
    parser.add_argument("--extra-cmd-topic", default=None)
    parser.add_argument("--command-timeout", type=float, default=0.25)
    parser.add_argument("--mode-machine", type=int, default=5)

    parser.add_argument("--left-hand-topic", default="rt/dex3/left/cmd")
    parser.add_argument("--right-hand-topic", default="rt/dex3/right/cmd")
    parser.add_argument("--left-hand-state-topic", default="rt/dex3/left/state")
    parser.add_argument("--right-hand-state-topic", default="rt/dex3/right/state")
    parser.add_argument("--hand-scale", type=float, default=1.0)
    parser.add_argument("--hand-offset", type=float, default=0.0)

    parser.add_argument("--lowstate-topic", default="rt/lowstate")
    parser.add_argument("--secondary-imu-topic", default="rt/secondary_imu")
    parser.add_argument("--odostate-topic", default="rt/odostate")
    parser.add_argument("--publish-log-interval", type=float, default=1.0)
    parser.add_argument("--command-log-interval", type=float, default=1.0)

    parser.add_argument("--hold-default-stand", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--publish-head-camera", action="store_true")
    parser.add_argument("--publish-wrist-cameras", action="store_true")
    parser.add_argument("--image-server-host", default="0.0.0.0")
    parser.add_argument("--image-config-port", type=int, default=60000)
    parser.add_argument("--head-camera-zmq-port", type=int, default=55555)
    parser.add_argument("--left-wrist-zmq-port", type=int, default=55556)
    parser.add_argument("--right-wrist-zmq-port", type=int, default=55557)
    parser.add_argument("--head-camera-webrtc-port", type=int, default=60001)
    parser.add_argument("--left-wrist-webrtc-port", type=int, default=60002)
    parser.add_argument("--right-wrist-webrtc-port", type=int, default=60003)
    parser.add_argument("--head-camera-name", default="robot0_agentview_center")
    parser.add_argument("--left-wrist-camera-name", default="robot0_agentview_left")
    parser.add_argument("--right-wrist-camera-name", default="robot0_agentview_right")
    parser.add_argument("--head-camera-height", type=int, default=480)
    parser.add_argument("--head-camera-width", type=int, default=640)
    parser.add_argument("--head-camera-fps", type=float, default=20.0)
    parser.add_argument("--head-camera-jpeg-quality", type=int, default=85)
    parser.add_argument("--show-camera-views", action="store_true")
    parser.add_argument("--show-overview-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overview-camera-distance", type=float, default=4.0)
    parser.add_argument("--overview-camera-azimuth", type=float, default=135.0)
    parser.add_argument("--overview-camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-view-height", type=int, default=480)
    parser.add_argument("--camera-view-width", type=int, default=640)
    parser.add_argument("--camera-view-fps", type=float, default=20.0)

    args = parser.parse_args()

    if args.camera is None:
        args.camera = "robot0_frontview"
    elif args.camera == "free":
        args.camera = None

    # Attributes consumed by the reused DDS receiver / camera publisher.
    args.subscribe_hands = True
    args.subscribe_gripper = False
    args.left_gripper_topic = "rt/dex1/left/cmd"
    args.right_gripper_topic = "rt/dex1/right/cmd"
    args.left_gripper_state_topic = "rt/dex1/left/state"
    args.right_gripper_state_topic = "rt/dex1/right/state"
    args.lower_body_mode = "off"
    args.lower_body_target_topic = "rt/g1/loco/joint_target"
    args.subscribe_run_command = False
    args.run_command_topic = "rt/run_command/cmd"
    args.hand_mapping_profile = "raw"
    args.hand_state_bootstrap_epsilon = 0.0
    args.gripper_open_value = 1.0
    return args


def main() -> None:
    args = parse_args()
    if args.rerender_episode:
        rerender_existing_episode(args)
        return

    print(
        colored(
            "Starting GROOT / SONIC DDS -> RoboCasa torque bridge: "
            f"task={args.task} robot={args.robot} domain={args.domain_id} "
            f"cmd_topic={args.cmd_topic} sim_frequency={args.sim_frequency:g}Hz",
            "yellow",
        )
    )
    robot_class = REGISTERED_ROBOTS.get(args.robot)
    if robot_class is None:
        print(colored(f"Resolved robot {args.robot}: not registered", "red"))
    else:
        print(
            colored(
                f"Resolved robot {args.robot}: {robot_class.__module__}.{robot_class.__name__}",
                "yellow",
            )
        )
    print(colored(f"Using GROOT / SONIC MJCF: {XML_PATH}", "yellow"))

    print(colored("Initializing Unitree DDS subscribers...", "yellow"))
    receiver = dds_utils.UnitreeDdsReceiver(args)
    env = make_env(args)
    install_enclosing_wall_hotkeys(env)
    env_info = json.dumps(
        {
            "env_name": args.task,
            "robots": args.robot,
            "controller_type": "sonic_direct_torque",
            "layout_ids": args.layout,
            "style_ids": args.style,
            "args": vars(args),
        }
    )
    collector = SonicRoboCasaDataCollector(args, env_info) if args.collect_demos else None
    collection_hotkeys = SonicCollectionHotkeys() if collector is not None else None
    publisher = GrootStatePublisher(args)
    image_publisher = (
        TeleimagerZmqMultiCameraPublisher(args)
        if args.publish_head_camera or args.publish_wrist_cameras
        else None
    )
    sim_viewer = None
    camera_viewer = None
    wall_hotkeys = None

    try:
        print(colored("Resetting RoboCasa scene. This can take a while on first load...", "yellow"))
        torque_map = reset_sonic_episode_env(env, args)
        wall_hotkeys = EnclosingWallHotkeyHandler(env)
        print(
            colored(
                f"G1Sonic torque map ready: {len(torque_map.joint_to_actuator)} joints, "
                f"nu={env.sim.model.nu}, action_dim={env.action_dim}, "
                f"cameras={env.sim.model.camera_names}",
                "green",
            )
        )

        render_main_viewer = args.render
        if render_main_viewer and args.show_camera_views:
            print(
                colored(
                    "Dual-view visualization enabled: MuJoCo viewer + 2x2 camera mosaic.",
                    "yellow",
                )
            )
        if render_main_viewer:
            sim_viewer = open_sim_viewer(env, args)
        if args.show_camera_views:
            camera_viewer = SimCameraMosaicViewer(env, args)

        if collector is not None and args.auto_start_recording:
            collector.start_episode(env)

        dt = 1.0 / args.sim_frequency
        next_command_log_time = 0.0
        next_render_time = 0.0
        while True:
            start = time.time()
            if collector is not None and collection_hotkeys is not None:
                if collection_hotkeys.consume_discard():
                    collector.finish_episode(env, discard=True)
                    print(colored("Resetting full RoboCasa scene after discarded episode...", "yellow"))
                    torque_map = reset_sonic_episode_env(env, args)
                    continue
                if collection_hotkeys.consume_stop():
                    collector.finish_episode(env)
                    print(colored("Resetting full RoboCasa scene after saved episode...", "yellow"))
                    torque_map = reset_sonic_episode_env(env, args)
                    continue
                if collection_hotkeys.consume_start():
                    if collector.recording:
                        print(colored("SONIC demo collection is already recording.", "yellow"))
                    else:
                        collector.start_episode(env)

            if wall_hotkeys is not None and wall_hotkeys.consume_pending(
                render=render_main_viewer and sim_viewer is None
            ):
                continue

            ctrl_action = apply_sonic_torques(env, torque_map, receiver.state, args)
            env.sim.step()
            if simulation_is_unstable(env):
                env.sim.data.ctrl[:] = 0.0
                reset_sonic_stand_state(env)
                continue
            env.update_state()
            if collector is not None and collector.record_step(env, ctrl_action, receiver.state):
                collector.finish_episode(env)
                print(colored("Resetting full RoboCasa scene after successful episode...", "yellow"))
                torque_map = reset_sonic_episode_env(env, args)
                continue
            publisher.publish(env)
            if image_publisher is not None:
                image_publisher.publish(env)

            now = time.monotonic()
            if args.command_log_interval > 0 and now >= next_command_log_time:
                log_sonic_command_state(receiver.state, args)
                next_command_log_time = now + args.command_log_interval
            if render_main_viewer and now >= next_render_time:
                if sim_viewer is None:
                    env.render()
                elif not sim_viewer.sync():
                    print(colored("MuJoCo passive viewer was closed.", "yellow"))
                    sim_viewer.close()
                    sim_viewer = None
                    render_main_viewer = False
                next_render_time = now + (1.0 / args.viewer_frequency)
            if camera_viewer is not None and not camera_viewer.sync(env):
                print(colored("Camera mosaic viewer was closed.", "yellow"))
                camera_viewer.close()
                camera_viewer = None

            elapsed = time.time() - start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print(colored("SONIC bridge stopped.", "yellow"))
    finally:
        if collector is not None:
            collector.close(env)
        if collection_hotkeys is not None:
            collection_hotkeys.close()
        if camera_viewer is not None:
            camera_viewer.close()
        if sim_viewer is not None:
            sim_viewer.close()
        receiver.stop()
        if image_publisher is not None:
            image_publisher.close()
        env.close()


if __name__ == "__main__":
    main()
