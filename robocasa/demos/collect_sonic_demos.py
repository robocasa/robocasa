"""Demo collection with the native robosuite SonicG1 path.

This is the RoboCasa-facing collection entry point for the robosuite SONIC branch.
Unlike the bring-up bridge, it keeps the normal robosuite / RoboCasa control and
dataset path intact:

* ``SonicG1`` is loaded from robosuite's robot registry.
* ``SONIC_WBC`` is used as the composite controller.
* simulation advances through ``env.step(action)``.
* states/actions are recorded by ``DataCollectionWrapper``.
* the resulting raw demo is packed with RoboCasa's existing hdf5 helper.
* optional videos are rendered by RoboCasa's existing playback utility.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import robosuite
import mujoco
from robosuite.controllers import load_composite_controller_config
import robosuite.controllers.composite.sonic_whole_body_controller as sonic_wbc
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING
import robosuite.utils.sonic.sources as sonic_sources
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored

import robocasa  # noqa: F401  registers RoboCasa tasks; robots come from robosuite's registry
from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallHotkeyHandler,
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)

try:
    from robosuite.scripts.collect_sonic_g1_demos import match_base_sim_physics
except Exception:  # pragma: no cover - only used when the external robosuite branch differs
    match_base_sim_physics = None


SONIC_CONTROLLER_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(robosuite.__file__))),
    "robosuite",
    "controllers",
    "config",
    "robots",
    "default_sonic_g1.json",
)

SONIC_FLOOR_FRICTION = 1.0
SONIC_FLOOR_TORSION = 0.005
SONIC_MUJOCO_SOLVER = mujoco.mjtSolver.mjSOL_NEWTON
SONIC_MUJOCO_SOLVER_ITERATIONS = 100
RESET_AFTER_SAVE = True
RECORD_CAMERA_VIDEOS = True
ENABLE_MUJOCO_VIEWER = True
MUJOCO_RENDERER = "mjviewer"
MUJOCO_RENDER_CAMERA = None
VIDEO_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
VIDEO_HEIGHT = 512
VIDEO_WIDTH = 768
IMAGE_OBS_HEIGHT = 128
IMAGE_OBS_WIDTH = 128
IMAGE_OBS_DONE_MODE = 2
IMAGE_OBS_NUM_PROCS = 1

_REAL_DDS_COMMAND_SOURCE = sonic_wbc.DDSCommandSource
_REAL_INIT_DDS_ONCE = sonic_sources.init_dds_once


def _gear_sonic_package_dir() -> Path | None:
    try:
        import gear_sonic  # noqa: PLC0415
    except ModuleNotFoundError:
        return None
    return Path(gear_sonic.__file__).resolve().parent


def _robosuite_sonic_robot_names() -> list[str]:
    return sorted(
        name
        for name in REGISTERED_ROBOTS
        if name in ROBOT_CLASS_MAPPING and name.startswith("Sonic")
    )


def _validate_robosuite_sonic_robot(robot: str) -> None:
    available = _robosuite_sonic_robot_names()
    if robot not in available:
        raise ValueError(
            f"Robot '{robot}' is not a SONIC robot registered by robosuite. "
            f"Available SONIC robosuite robots: {', '.join(available) or '<none>'}."
        )


def _find_with_hand_xml() -> str | None:
    env_path = os.environ.get("SONIC_G1_XML")
    if env_path:
        return env_path

    gear_sonic_dir = _gear_sonic_package_dir()
    if gear_sonic_dir is None:
        return None

    candidates = (
        gear_sonic_dir / "data" / "robots" / "g1" / "g1_29dof_with_hand.xml",
        gear_sonic_dir / "data" / "robot_model" / "model_data" / "g1" / "g1_29dof_with_hand.xml",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _default_body_command_from_config(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(config["NUM_MOTORS"])
    q = np.asarray(
        config.get("DEFAULT_MOTOR_ANGLES", config.get("DEFAULT_DOF_ANGLES")),
        dtype=float,
    )[:n]
    kp = np.asarray(config["MOTOR_KP"], dtype=float)[:n]
    kd = np.asarray(config["MOTOR_KD"], dtype=float)[:n]
    return q, kp, kd


def _install_configured_dds_source(domain_id: int, network_interface: str | None) -> None:
    """Patch SONIC_WBC's DDS source factory with runtime DDS settings.

    ``SonicWholeBodyController`` loads the gear_sonic WBC yaml internally, so the
    network settings do not naturally flow through robosuite's controller config.
    This mirrors robosuite's SONIC collection script source patching, but keeps the
    source as live DDS instead of replacing it with replay/mock data.
    """

    def _init_dds_once(config: dict[str, Any]) -> None:
        with sonic_sources._dds_lock:
            if sonic_sources._dds_initialized:
                return
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize

            if config.get("INTERFACE"):
                print(
                    colored(
                        f"Initializing SONIC DDS: domain={config['DOMAIN_ID']} interface={config['INTERFACE']}",
                        "yellow",
                    )
                )
                ChannelFactoryInitialize(config["DOMAIN_ID"], networkInterface=config["INTERFACE"])
            else:
                print(colored(f"Initializing SONIC DDS: domain={config['DOMAIN_ID']} interface=<default>", "yellow"))
                ChannelFactoryInitialize(config["DOMAIN_ID"])
            sonic_sources._dds_initialized = True

    def _source(config: dict[str, Any]):
        config = dict(config)
        config["DOMAIN_ID"] = int(domain_id)
        if network_interface:
            config["INTERFACE"] = network_interface
        hold_q, hold_kp, hold_kd = _default_body_command_from_config(config)
        return _REAL_DDS_COMMAND_SOURCE(
            config,
            hold_q=hold_q,
            hold_kp=hold_kp,
            hold_kd=hold_kd,
        )

    sonic_sources.init_dds_once = _init_dds_once
    sonic_wbc.DDSCommandSource = _source


def _install_mock_dds_source_for_playback() -> None:
    """Make playback_dataset_hdf5 instantiate SONIC_WBC without opening DDS."""

    from robosuite.utils.sonic.sources import ReferenceMockSource

    def _source(config: dict[str, Any]):
        n = int(config["NUM_MOTORS"])
        q, kp, kd = _default_body_command_from_config(config)
        return ReferenceMockSource(q.reshape(1, n), kp, kd)

    sonic_wbc.DDSCommandSource = _source


def _restore_dds_source() -> None:
    sonic_sources.init_dds_once = _REAL_INIT_DDS_ONCE
    sonic_wbc.DDSCommandSource = _REAL_DDS_COMMAND_SOURCE


def _sonic_command_action(env) -> np.ndarray:
    """Pack the latest SONIC motor targets into the standard action dataset slot."""

    out = np.zeros(env.action_dim)
    controller = env.robots[0].composite_controller
    last = getattr(controller, "last_command", None)
    if last is None or last[0] is None:
        return out

    cmd, hands = last
    parts = [np.asarray(cmd.q, dtype=float)]
    if hands is not None:
        parts.extend([np.asarray(hands[0].q, dtype=float), np.asarray(hands[1].q, dtype=float)])
    action = np.concatenate(parts)
    out[: min(out.shape[0], action.shape[0])] = action[: out.shape[0]]
    return out


def _sonic_command_metadata(env) -> dict[str, np.ndarray]:
    controller = env.robots[0].composite_controller
    last = getattr(controller, "last_command", None)
    if last is None or last[0] is None:
        return {}
    cmd, hands = last
    info = {
        "sonic_body_q": np.asarray(cmd.q),
        "sonic_body_dq": np.asarray(cmd.dq),
        "sonic_body_kp": np.asarray(cmd.kp),
        "sonic_body_kd": np.asarray(cmd.kd),
        "sonic_body_tau": np.asarray(cmd.tau),
    }
    if hands is not None:
        left, right = hands
        for prefix, hand_cmd in (("sonic_left_hand", left), ("sonic_right_hand", right)):
            info[f"{prefix}_q"] = np.asarray(hand_cmd.q)
            info[f"{prefix}_dq"] = np.asarray(hand_cmd.dq)
            info[f"{prefix}_kp"] = np.asarray(hand_cmd.kp)
            info[f"{prefix}_kd"] = np.asarray(hand_cmd.kd)
            info[f"{prefix}_tau"] = np.asarray(hand_cmd.tau)
    return info


class SonicDataCollectionWrapper(DataCollectionWrapper):
    """Small extension that keeps the standard wrapper as the source of truth.

    ``SONIC_WBC`` ignores the input action and reads live DDS during controller
    execution. After the step, the wrapper replaces the recorded action with the
    current SONIC motor target so the hdf5 action slot contains meaningful commands.
    Extra SONIC fields remain optional metadata in the raw ``state_*.npz`` files.
    """

    def step(self, action):
        ret = super().step(action)
        if self.action_infos:
            self.action_infos[-1]["actions"] = _sonic_command_action(self.env)
            self.action_infos[-1].update(_sonic_command_metadata(self.env))
        return ret

    def start_episode_from_current_state(self):
        """Start data collection from the already-running simulation state.

        The SONIC demo loop runs MuJoCo before recording so the external controller
        can receive lowstate and stabilize the robot. Pressing ``r`` should only
        start logging, not reset the task.
        """

        if self.has_interaction:
            self._flush()
        self.t = 0
        self.states = []
        self.action_infos = []
        self.has_interaction = False
        self.ep_directory = None
        self._current_task_instance_xml = (
            self.env.model.get_xml()
            if self.use_env_xml_for_reset
            else self.env.sim.model.get_xml()
        )
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())
        self.env.set_ep_meta(self.env.get_ep_meta())


@dataclass
class CollectionState:
    recording: bool = False
    successful_episodes: list[str] | None = None
    last_status_log_time: float = 0.0
    last_body_command_q: np.ndarray | None = None

    def __post_init__(self):
        if self.successful_episodes is None:
            self.successful_episodes = []


class SonicCollectionHotkeys:
    def __init__(self):
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        try:
            from pynput.keyboard import Listener
        except ImportError:
            print(colored("pynput is not installed; hotkeys are disabled.", "yellow"))
            return

        def _on_release(key):
            char = getattr(key, "char", None)
            if char in {"r", "q", "x"}:
                with self._lock:
                    self._pending.add(char)

        self._listener = Listener(on_release=_on_release)
        self._listener.start()
        print(colored("SONIC demo hotkeys: r=start, q=save, x=discard, Ctrl+C=exit.", "yellow"))

    def consume(self) -> set[str]:
        with self._lock:
            pending = set(self._pending)
            self._pending.clear()
        return pending

    def close(self):
        if self._listener is not None:
            self._listener.stop()


def make_env(args, controller_config):
    config = dict(
        env_name=args.task,
        robots=args.robot,
        controller_configs=controller_config,
        initialization_noise=None,
        layout_ids=args.layout,
        style_ids=args.style,
        translucent_robot=not args.no_translucent_robot,
    )

    env = robosuite.make(
        **config,
        has_renderer=ENABLE_MUJOCO_VIEWER,
        has_offscreen_renderer=False,
        render_camera=MUJOCO_RENDER_CAMERA,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=args.horizon,
        renderer=MUJOCO_RENDERER,
    )
    env = VisualizationWrapper(env)
    for setting in ("robots", "grippers"):
        if setting in env.get_visualization_settings():
            env.set_visualization_setting(setting, False)
    env.env.visualize(vis_settings=env._vis_settings)
    env = EnclosingWallRenderWrapper(env, alpha=0.15, enabled=False)
    install_enclosing_wall_hotkeys(env)
    return env, config


def _apply_sonic_physics(env) -> None:
    if match_base_sim_physics is None:
        return
    match_base_sim_physics(
        env.sim.model._model,
        floor_friction=SONIC_FLOOR_FRICTION,
        floor_torsion=SONIC_FLOOR_TORSION,
    )


def _configure_mujoco_solver(env) -> None:
    """Use a robust solver for dense robot / kitchen contacts during teleop."""

    opt = env.sim.model._model.opt
    opt.solver = int(SONIC_MUJOCO_SOLVER)
    opt.iterations = SONIC_MUJOCO_SOLVER_ITERATIONS


def _configure_sonic_after_reset(env) -> None:
    _apply_sonic_physics(env)
    _configure_mujoco_solver(env)
    controller = getattr(env.robots[0], "composite_controller", None)
    if controller is not None and hasattr(controller, "release_band"):
        controller.release_band()


def _sonic_controller_source_bridge(env):
    controller = getattr(env.robots[0], "composite_controller", None)
    source = getattr(controller, "_src", None)
    bridge = getattr(source, "bridge", None)
    return controller, source, bridge


def _render_viewer(env) -> None:
    """Refresh the on-screen robosuite viewer without advancing physics."""

    viewer = getattr(env, "viewer", None)
    if viewer is not None and hasattr(viewer, "update"):
        viewer.update()
        return
    if hasattr(env, "render"):
        env.render()


def _install_viewer_update_throttle(env, args) -> None:
    """Throttle robosuite's per-step passive viewer sync.

    robosuite updates the viewer inside every ``env.step``. At 500 Hz this makes
    the GUI thread the pacing bottleneck, so we keep physics / DDS at control
    frequency while syncing the passive viewer at a lower rate.
    """

    if args.viewer_update_freq <= 0:
        return

    viewer = getattr(env, "viewer", None)
    if viewer is None or not hasattr(viewer, "update"):
        return
    if getattr(viewer, "_sonic_update_throttled", False):
        return

    original_update = viewer.update
    viewer._sonic_update_throttled = True
    viewer._sonic_original_update = original_update
    viewer._sonic_next_update_time = 0.0
    viewer._sonic_update_period = 1.0 / float(args.viewer_update_freq)

    def throttled_update():
        now = time.time()
        if now < viewer._sonic_next_update_time:
            return
        viewer._sonic_next_update_time = now + viewer._sonic_update_period
        return original_update()

    viewer.update = throttled_update


def _sonic_status(env, state: CollectionState) -> dict[str, Any]:
    controller, source, bridge = _sonic_controller_source_bridge(env)
    status: dict[str, Any] = {
        "source_ready": source is not None,
        "bridge_ready": bridge is not None,
        "lowcmd": False,
        "left_hand": False,
        "right_hand": False,
        "body_q_delta": None,
    }
    if bridge is not None:
        status["lowcmd"] = bool(getattr(bridge, "low_cmd_received", False))
        status["left_hand"] = bool(getattr(bridge, "left_hand_cmd_received", False))
        status["right_hand"] = bool(getattr(bridge, "right_hand_cmd_received", False))

    last = getattr(controller, "last_command", None) if controller is not None else None
    if last is not None and last[0] is not None:
        body_q = np.asarray(last[0].q, dtype=float)
        if state.last_body_command_q is not None and state.last_body_command_q.shape == body_q.shape:
            status["body_q_delta"] = float(np.max(np.abs(body_q - state.last_body_command_q)))
        state.last_body_command_q = body_q.copy()
    return status


def _maybe_log_sonic_status(env, state: CollectionState, args) -> None:
    if args.status_log_interval <= 0:
        return
    now = time.time()
    if now - state.last_status_log_time < args.status_log_interval:
        return
    state.last_status_log_time = now
    status = _sonic_status(env, state)
    delta = status["body_q_delta"]
    delta_text = "n/a" if delta is None else f"{delta:.3e}"
    print(
        colored(
            "SONIC DDS status: "
            f"source={status['source_ready']} "
            f"bridge={status['bridge_ready']} "
            f"lowcmd={status['lowcmd']} "
            f"left_hand={status['left_hand']} "
            f"right_hand={status['right_hand']} "
            f"body_q_delta={delta_text}",
            "cyan",
        )
    )


def _flush_current_episode(env: SonicDataCollectionWrapper) -> str | None:
    if not env.has_interaction or env.ep_directory is None:
        return None
    ep_name = os.path.basename(env.ep_directory)
    if len(env.states) > 0:
        env._flush()
    env.has_interaction = False
    env.states = []
    env.action_infos = []
    return ep_name


def _write_episode_metadata(ep_directory: str, env_info: str, success: bool) -> None:
    with open(os.path.join(ep_directory, "ep_stats.json"), "w") as f:
        json.dump({"success": success}, f)
    with open(os.path.join(ep_directory, "env_info.json"), "w") as f:
        json.dump(env_info, f)


def _pack_dataset(all_eps_directory: str, demo_dir: str, env_info: str, successful_episodes: list[str]) -> str | None:
    hdf5_path = gather_demonstrations_as_hdf5(
        all_eps_directory,
        demo_dir,
        env_info,
        successful_episodes=successful_episodes,
        verbose=True,
    )
    if hdf5_path is not None:
        convert_to_robomimic_format(hdf5_path)
    return hdf5_path


def _render_dataset_video(hdf5_path: str, args) -> None:
    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import playback_dataset

    video_path = os.path.splitext(hdf5_path)[0] + ".mp4"
    print(colored(f"Rendering dataset video with RoboCasa playback: {video_path}", "yellow"))
    _install_mock_dds_source_for_playback()
    try:
        playback_dataset(
            dataset=hdf5_path,
            use_actions=False,
            use_abs_actions=False,
            use_obs=False,
            filter_key=None,
            n=None,
            render=False,
            render_image_names=VIDEO_CAMERAS,
            camera_height=VIDEO_HEIGHT,
            camera_width=VIDEO_WIDTH,
            video_path=video_path,
            video_skip=args.video_skip,
            extend_states=False,
            first=False,
            verbose=args.verbose,
        )
    finally:
        _install_configured_dds_source(args.domain_id, args.network_interface)


def _extract_image_observations(hdf5_path: str, args) -> str | None:
    from robocasa.scripts.dataset_scripts.dataset_states_to_obs import dataset_states_to_obs_mp

    print(colored(f"Extracting image observations with RoboCasa dataset_states_to_obs: {hdf5_path}", "yellow"))
    _install_mock_dds_source_for_playback()
    try:
        extract_args = argparse.Namespace(
            dataset=hdf5_path,
            output_name=None,
            n=None,
            shaped=False,
            camera_names=VIDEO_CAMERAS,
            camera_height=IMAGE_OBS_HEIGHT,
            camera_width=IMAGE_OBS_WIDTH,
            done_mode=IMAGE_OBS_DONE_MODE,
            copy_rewards=False,
            copy_dones=False,
            include_next_obs=False,
            no_compress=False,
            generative_textures=False,
            randomize_cameras=False,
            num_procs=IMAGE_OBS_NUM_PROCS,
            gpu_ids=None,
            procs_per_gpu=None,
        )
        stats = dataset_states_to_obs_mp(extract_args)
    finally:
        _install_configured_dds_source(args.domain_id, args.network_interface)

    output_path = stats.get("name") if stats else None
    if output_path is not None:
        print(colored(f"Extracted SONIC image-observation dataset: {output_path}", "green"))
    return output_path


def _finalize_dataset_outputs(
    all_eps_directory: str,
    demo_dir: str,
    env_info: str,
    successful_episodes: list[str],
    args,
) -> str | None:
    hdf5_path = _pack_dataset(all_eps_directory, demo_dir, env_info, successful_episodes)
    if hdf5_path is None:
        return None
    _extract_image_observations(hdf5_path, args)
    if RECORD_CAMERA_VIDEOS:
        _render_dataset_video(hdf5_path, args)
    return hdf5_path


def _finish_episode(env, args, state, all_eps_directory: str, demo_dir: str, env_info: str, discard: bool) -> None:
    ep_name = _flush_current_episode(env)
    if ep_name is None:
        print(colored("No active episode data to finish.", "yellow"))
        state.recording = False
        return

    ep_directory = os.path.join(all_eps_directory, ep_name)
    success = not discard
    _write_episode_metadata(ep_directory, env_info, success)
    if success:
        state.successful_episodes.append(ep_name)

    gather_demonstrations_as_hdf5(
        all_eps_directory,
        ep_directory,
        env_info,
        successful_episodes=[ep_name],
        out_name="ep_demo.hdf5",
    )

    _finalize_dataset_outputs(
        all_eps_directory,
        demo_dir,
        env_info,
        state.successful_episodes,
        args,
    )

    state.recording = False
    print(colored(f"Saved SONIC episode: {ep_directory}", "green"))

    if RESET_AFTER_SAVE:
        print(colored("Resetting RoboCasa scene for the next episode.", "yellow"))
        env.reset()
        _configure_sonic_after_reset(env.env)
        _install_viewer_update_throttle(env.env, args)


def run_collection(args) -> None:
    _validate_robosuite_sonic_robot(args.robot)

    with_hand_xml = args.sonic_g1_xml or _find_with_hand_xml()
    if with_hand_xml:
        os.environ.setdefault("SONIC_G1_XML", with_hand_xml)
    else:
        print(colored("SONIC_G1_XML was not found; Dex3 DDS hand topics may be disabled.", "yellow"))

    _install_configured_dds_source(args.domain_id, args.network_interface)
    controller_config = load_composite_controller_config(controller=SONIC_CONTROLLER_CONFIG)
    base_env, env_config = make_env(args, controller_config)

    time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
    demo_dir = os.path.join(args.directory, f"{time_str}_{args.task}_sonic")
    all_eps_directory = os.path.join(demo_dir, "episodes")
    os.makedirs(all_eps_directory, exist_ok=True)

    env_info = json.dumps(env_config)
    env = SonicDataCollectionWrapper(
        base_env,
        all_eps_directory,
        collect_freq=args.collect_freq,
        flush_freq=args.flush_freq,
        use_env_xml_for_reset=True,
    )

    hotkeys = SonicCollectionHotkeys()
    wall_hotkeys = EnclosingWallHotkeyHandler(base_env)
    state = CollectionState()
    zero_action = np.zeros(env.action_dim)

    print(colored(f"Dataset directory: {demo_dir}", "green"))
    print(colored("Start the SONIC policy separately. Simulation is live; press r to start recording.", "yellow"))

    try:
        base_env.reset()
        _configure_sonic_after_reset(base_env)
        _render_viewer(base_env)
        _install_viewer_update_throttle(base_env, args)
        while True:
            pending = hotkeys.consume()
            if wall_hotkeys.consume_pending(render=True):
                continue

            if "r" in pending and not state.recording:
                env.start_episode_from_current_state()
                state.recording = True
                state.last_status_log_time = 0.0
                state.last_body_command_q = None
                print(colored("Started recording SONIC episode.", "green"))

            if "q" in pending and state.recording:
                _finish_episode(
                    env,
                    args,
                    state,
                    all_eps_directory,
                    demo_dir,
                    env_info,
                    discard=False,
                )
                continue

            if "x" in pending and state.recording:
                _finish_episode(
                    env,
                    args,
                    state,
                    all_eps_directory,
                    demo_dir,
                    env_info,
                    discard=True,
                )
                continue

            if state.recording:
                env.step(zero_action)
                _maybe_log_sonic_status(base_env, state, args)
            else:
                base_env.step(zero_action)
                _maybe_log_sonic_status(base_env, state, args)

    except KeyboardInterrupt:
        print()
        if state.recording:
            _finish_episode(
                env,
                args,
                state,
                all_eps_directory,
                demo_dir,
                env_info,
                discard=False,
            )
        else:
            _finalize_dataset_outputs(
                all_eps_directory,
                demo_dir,
                env_info,
                state.successful_episodes,
                args,
            )
    finally:
        hotkeys.close()
        env.close()
        _restore_dds_source()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="CoffeeSetupMug", help="RoboCasa task / environment name")
    parser.add_argument(
        "--robot",
        default="SonicG1",
        help="SONIC robot name from robosuite's robot registry, e.g. SonicG1.",
    )
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--directory", default="./datasets")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--network-interface", default=None, help="DDS network interface, e.g. wlp6s0")
    parser.add_argument(
        "--sonic-g1-xml",
        default=None,
        help="Optional with-hand SONIC XML override. Defaults to SONIC_G1_XML or the installed gear_sonic package.",
    )
    parser.add_argument("--control-freq", type=int, default=500)
    parser.add_argument("--viewer-update-freq", type=float, default=30.0)
    parser.add_argument("--status-log-interval", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=10_000_000)
    parser.add_argument("--collect-freq", type=int, default=1)
    parser.add_argument("--flush-freq", type=int, default=1_000_000_000)
    parser.add_argument("--no-translucent-robot", action="store_true")
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    run_collection(args)


if __name__ == "__main__":
    main()
