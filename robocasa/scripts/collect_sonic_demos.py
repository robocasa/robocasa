"""Collect SONIC G1 demos in robocasa kitchens over live DDS.

The env action is the per-motor q* SONIC command read from DDS; SonicWholeBodyController applies the
PD law with the (constant) gains captured from the stream. Episodes are recorded in standard
robocasa/robomimic format (SONIC_WBC kept as the env_info controller, so offline playback /
obs-extraction work without DDS). Multi-episode hotkeys (typed in the terminal; chosen to avoid the
SONIC controller's reserved keys): c=record, k=save (keep), x=discard, b=toggle band.

Start the C++ SONIC controller (publishes lowcmd to DDS) first, then run this on a machine with a
display (do NOT set MUJOCO_GL=egl).
"""
import argparse
from copy import deepcopy
import datetime
import json
import os
import threading
import time

# Editable RoboSuite checkouts do not always provide Numba with a source-cache
# locator. Keep compiled artifacts outside the repository and isolate users.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", f"/tmp/robocasa-numba-cache-{os.getuid()}"
)

import h5py
import mujoco
import numpy as np

from termcolor import colored

import robosuite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper
from robosuite.scripts.collect_sonic_g1_demos import match_base_sim_physics, SONIC_CFG
from robosuite.utils.sonic.action_sources import DDSActionSource

import robocasa  # noqa: F401  registers the kitchen envs
from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper


# PGS avoids Newton's sparse Hessian factorization failures under dense contacts.
SONIC_MUJOCO_SOLVER = "PGS"


def _controller(base):
    # recreated on every env.reset() (robot.reset -> _load_controller) -> always fetch fresh
    return base.robots[0].composite_controller


def _snapshot_and_print_instruction(base):
    """Capture the sampled episode metadata exactly once and announce its goal."""
    ep_meta = base.get_ep_meta()
    base.set_ep_meta(ep_meta)
    lang = ep_meta.get("lang")
    instruction = lang.strip() if isinstance(lang, str) and lang.strip() else None
    if instruction:
        print(colored(f"Instruction: {instruction}", "green"), flush=True)
    return ep_meta, instruction


def make_env(args, cfg):
    """Build the robocasa kitchen env + the JSON-able env_kwargs recorded into the dataset (env_name
    is stored separately, per the robomimic env_args convention). SONIC_WBC is kept as the
    controller -- it opens no DDS at construction. Uses robosuite's built-in mjviewer (throttled via
    base.render_freq, re-created on each reset)."""
    env = robosuite.make(
        args.environment, robots=[args.robot], controller_configs=cfg,
        has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
        renderer="mjviewer", render_camera=args.render_camera,
        layout_ids=args.layout, style_ids=args.style, control_freq=args.control_freq,
        initialization_noise=None, mujoco_solver=SONIC_MUJOCO_SOLVER,
    )
    env_kwargs = dict(robots=[args.robot], controller_configs=cfg, initialization_noise=None,
                      use_camera_obs=False, translucent_robot=False,
                      layout_ids=args.layout, style_ids=args.style, control_freq=args.control_freq,
                      mujoco_solver=SONIC_MUJOCO_SOLVER)
    return env, env_kwargs


def _apply_runtime(base, args):
    # match SONIC base_sim physics + throttle the heavy per-step bookkeeping / viewer off the
    # control loop. Re-applied after each reset (hard reset rebuilds the model).
    match_base_sim_physics(base.sim.model._model, args.floor_friction, args.floor_torsion,
                           timestep=args.sim_dt)
    base.sim.model._model.opt.solver = int(mujoco.mjtSolver.mjSOL_PGS)
    base.post_action_freq = max(1, round(args.control_freq / args.post_action_hz))
    base.render_freq = max(1, round(args.control_freq / args.render_hz))


def reset_with_retry(env, base, args, tries=12):
    # Keep a reset fallback for invalid placements and other fatal simulation errors.
    last = None
    reset_started = time.perf_counter()
    for k in range(tries):
        attempt_started = time.perf_counter()
        # Episode capture serializes fixture refs; fresh samples must not reuse that metadata.
        base.unset_ep_meta()
        try:
            ret = env.reset()
            _apply_runtime(base, args)
            print(
                f"[sonic-timing] env.reset ready in "
                f"{time.perf_counter() - reset_started:.3f}s "
                f"(attempt {k + 1}, current attempt "
                f"{time.perf_counter() - attempt_started:.3f}s)",
                flush=True,
            )
            return ret
        except mujoco.FatalError as e:
            last = e
            print(
                f"[sonic] reset solver error after "
                f"{time.perf_counter() - attempt_started:.3f}s; "
                f"re-sampling ({k + 2}/{tries})",
                flush=True,
            )
    raise last


def _gains_json(gains):
    if not gains:
        return None
    return json.dumps({k: [np.asarray(kp).tolist(), np.asarray(kd).tolist()]
                       for k, (kp, kd) in gains.items()})


def _valid_sonic_gains(gains):
    if not gains or "body" not in gains:
        return False
    try:
        return np.any(np.abs(np.asarray(gains["body"][0], dtype=float)) > 1e-9)
    except Exception:
        return False


def _copy_sonic_gains(gains):
    return {k: (np.asarray(kp, dtype=float).copy(), np.asarray(kd, dtype=float).copy())
            for k, (kp, kd) in gains.items()}


def _resolve_episode_event(pressed, vr_events, recording):
    """Merge local and VR lifecycle events with deterministic abort-first priority."""
    local_abort = "x" in pressed
    vr_abort = "toggle_data_abort" in vr_events
    if local_abort or vr_abort:
        return ("discard", vr_abort) if recording else (None, vr_abort)

    vr_toggle = "toggle_data_collection" in vr_events
    if recording and ("k" in pressed or vr_toggle):
        return "save", vr_toggle
    if not recording and ("c" in pressed or vr_toggle):
        return "start", vr_toggle
    return None, False


def _sonic_runtime_json(args):
    post_action_freq = max(1, round(args.control_freq / args.post_action_hz))
    render_freq = max(1, round(args.control_freq / args.render_hz))
    return json.dumps({
        "sim_dt": float(args.sim_dt),
        "control_freq": int(args.control_freq),
        "post_action_hz": float(args.post_action_hz),
        "post_action_freq": int(post_action_freq),
        "render_hz": float(args.render_hz),
        "render_freq": int(render_freq),
        "floor_friction": float(args.floor_friction),
        "floor_torsion": float(args.floor_torsion),
        "mujoco_solver": SONIC_MUJOCO_SOLVER,
        "mujoco_state_spec": "mjSTATE_INTEGRATION",
    })


def _stamp_sonic_attrs(path, gains, args):
    # Gains/runtime are constant over a session -> dataset-level attrs; replay restores them.
    gj = _gains_json(gains)
    if path:
        with h5py.File(path, "a") as f:
            if "data" in f:
                if gj:
                    f["data"].attrs["sonic_gains"] = gj
                f["data"].attrs["sonic_runtime"] = _sonic_runtime_json(args)


class Hotkeys:
    """Global c/k/x/b hotkeys (pynput, non-blocking). Disabled gracefully if pynput is missing.
    Keys avoid the SONIC controller's reserved set (R/Q/etc.) since this listener is global."""

    KEYS = {"c", "k", "x", "b"}

    def __init__(self):
        self._pending, self._lock, self._listener = set(), threading.Lock(), None
        try:
            from pynput.keyboard import Listener
        except ImportError:
            print("[sonic] pynput missing; hotkeys disabled.", flush=True)
            return

        def on_release(key):
            c = getattr(key, "char", None)
            if c in self.KEYS:
                with self._lock:
                    self._pending.add(c)

        self._listener = Listener(on_release=on_release)
        self._listener.start()
        print("[sonic] hotkeys: c=record k=save x=discard b=band Ctrl-C=exit", flush=True)

    def consume(self):
        with self._lock:
            p = set(self._pending)
            self._pending.clear()
        return p

    def close(self):
        if self._listener:
            self._listener.stop()


class SonicDataCollectionWrapper(DataCollectionWrapper):
    """Records the q* action as-is (it is the meaningful env action) and begins logging from the
    already-running, DDS-stabilised sim. Skips the per-step _check_success (decided by q/x, and it
    is the cost we throttle via base.post_action_freq).

    In addition to the standard flattened MuJoCo state (time/qpos/qvel/act), SONIC stores
    mjSTATE_INTEGRATION. Reactive walking can amplify solver warm-start differences through
    near-grazing hand/foot contacts, so future replays need the richer initial integration state.
    """

    @staticmethod
    def _integration_state_for(env):
        model = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
        data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=float)
        mujoco.mj_getState(model, data, state, spec)
        return state

    def _on_first_interaction(self):
        # DataCollectionWrapper calls env.get_ep_meta() here. Use the exact
        # snapshot already published to the exporter instead, because novel
        # instruction generators may otherwise sample a second string.
        self.has_interaction = True
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, f"ep_{t1}_{t2}")
        assert not os.path.exists(self.ep_directory)
        print(f"DataCollectionWrapper: making folder at {self.ep_directory}")
        os.makedirs(self.ep_directory)

        with open(os.path.join(self.ep_directory, "model.xml"), "w", encoding="utf-8") as f:
            f.write(self._current_task_instance_xml)
        with open(os.path.join(self.ep_directory, "ep_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self._current_task_instance_ep_meta, f)

        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)
        self.integration_states.append(self._current_task_instance_integration_state)

    def _flush(self):
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, f"state_{t1}_{t2}.npz")
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            integration_states=np.array(self.integration_states),
            action_infos=self.action_infos,
            successful=self.successful,
            env=env_name,
        )
        self.states = []
        self.integration_states = []
        self.action_infos = []
        self.successful = False

    def step(self, action):
        ret = self.env.step(action)
        self.t += 1
        if not self.has_interaction:
            self._on_first_interaction()
        if self.t % self.collect_freq == 0:
            self.states.append(self.env.sim.get_state().flatten())
            self.integration_states.append(self._integration_state_for(self.env))
            self.action_infos.append({"actions": np.asarray(action, dtype=float)})
        return ret

    def start_episode_from_current_state(self, ep_meta=None):
        if self.has_interaction:
            self._flush()
        self.t = 0
        self.states = []
        self.integration_states = []
        self.action_infos = []
        self.has_interaction = False
        self.ep_directory = None
        self._current_task_instance_xml = self.env.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())
        self._current_task_instance_integration_state = self._integration_state_for(self.env)
        # Reuse the metadata snapshot already sent to the VLA exporter. Some
        # novel-instruction tasks sample text inside get_ep_meta(), so reading it
        # again here could give the raw demo and LeRobot export different labels.
        selected_ep_meta = ep_meta if ep_meta is not None else self.env.get_ep_meta()
        self._current_task_instance_ep_meta = deepcopy(selected_ep_meta)
        self.env.set_ep_meta(deepcopy(selected_ep_meta))


def run_collection(args, base, wall, env_kwargs, source):
    """Live multi-episode DDS collection. Owns the MuJoCo clock; writes standard-format episodes."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    demo_dir = os.path.join(args.out, f"{ts}_{args.environment}_sonic")
    eps = os.path.join(demo_dir, "episodes")
    os.makedirs(eps, exist_ok=True)
    env_info = json.dumps(env_kwargs)
    env = SonicDataCollectionWrapper(wall, eps, collect_freq=args.record_freq, flush_freq=0,
                                     use_env_xml_for_reset=True)

    manager_sub = None
    keyboard_pub = None
    instruction_pub = None
    image_pub = None
    if args.vla_stream:
        from robocasa.utils.sonic_vla_streaming import (
            ManagerStateSubscriber,
            RoboCasaVLACameraPublisher,
            VLACameraConfig,
            VLAExporterKeyboardPublisher,
            VLAExporterInstructionPublisher,
        )

        image_pub = RoboCasaVLACameraPublisher(
            VLACameraConfig(
                camera_name=args.vla_camera_name,
                output_key=args.vla_camera_key,
                camera_names=args.vla_camera_names,
                output_keys=args.vla_camera_keys,
                width=args.vla_camera_width,
                height=args.vla_camera_height,
                hz=args.vla_camera_hz,
                port=args.vla_camera_port,
                flip_vertical=args.vla_camera_flip,
            )
        )
        image_pub.start(base)
        manager_sub = ManagerStateSubscriber(args.vla_manager_host, args.vla_manager_port)
        if args.vla_keyboard_sync:
            keyboard_pub = VLAExporterKeyboardPublisher(args.vla_keyboard_port)
        if args.vla_instruction_sync:
            instruction_pub = VLAExporterInstructionPublisher(
                args.vla_instruction_port,
                repeat_interval=args.vla_instruction_heartbeat,
            )
        print(
            f"[sonic-vla] publishing {','.join(args.vla_camera_keys)} "
            f"from {','.join(args.vla_camera_names)} "
            f"at {args.vla_camera_hz:g} Hz on port {args.vla_camera_port}",
            flush=True,
        )
    keys = Hotkeys()
    hold = np.zeros(base.action_dim)
    saved, recording, gains, saved_gains = [], False, None, None
    next_t = time.perf_counter()
    instruction_sequence = 0
    current_ep_meta = None
    current_instruction = None
    print(f"[sonic] dataset dir: {demo_dir}", flush=True)
    print("[sonic] press 'b' to drop the band once balancing, then 'c' to record.", flush=True)

    def announce_episode():
        nonlocal current_ep_meta, current_instruction, instruction_sequence
        current_ep_meta, current_instruction = _snapshot_and_print_instruction(base)
        if instruction_pub is not None:
            episode_id = f"{ts}:{instruction_sequence:06d}"
            instruction_pub.set_episode(
                current_instruction,
                episode_id=episode_id,
                sequence=instruction_sequence,
            )
            source_label = current_instruction or "<CLI --task-prompt fallback>"
            print(
                f"[sonic-vla] episode instruction {episode_id} -> {source_label}",
                flush=True,
            )
        instruction_sequence += 1

    announce_episode()

    def sync_exporter(key, from_vr=False, delay=True):
        if not args.vla_stream:
            return
        if not from_vr and keyboard_pub is not None:
            keyboard_pub.send(key)
        if delay and key in {"c", "x"} and args.vla_save_sync_delay > 0:
            time.sleep(args.vla_save_sync_delay)

    def finish(discard):
        nonlocal recording, saved_gains, next_t
        ep = env.ep_directory
        if not env.has_interaction or ep is None:
            recording = False
            return
        transition_started = time.perf_counter()
        name = os.path.basename(ep)
        if env.states:
            env._flush()
        env.has_interaction, env.states, env.integration_states, env.action_infos = False, [], [], []
        recording = False
        if discard:
            print(f"[sonic] discarded {name}", flush=True)
        else:
            saved.append(name)
            if _valid_sonic_gains(gains):
                saved_gains = _copy_sonic_gains(gains)
            h = gather_demonstrations_as_hdf5(eps, ep, env_info, successful_episodes=[name],
                                              out_name="ep_demo.hdf5")
            _stamp_sonic_attrs(h, saved_gains or gains, args)
            if h:
                convert_to_robomimic_format(h, filter_num_demos=None)
            print(f"[sonic] saved {name}", flush=True)
        finalize_finished = time.perf_counter()
        reset_with_retry(wall, base, args)
        reset_finished = time.perf_counter()
        source.reset(base)
        announce_episode()
        ready_at = time.perf_counter()
        # A hard reset and optional HDF5 conversion can take seconds. Do not
        # make the 200 Hz loop chase those expired wall-clock deadlines.
        next_t = ready_at
        print(
            f"[sonic-timing] episode {'discard' if discard else 'save'} transition: "
            f"finalize={finalize_finished - transition_started:.3f}s, "
            f"reset={reset_finished - finalize_finished:.3f}s, "
            f"source+instruction={ready_at - reset_finished:.3f}s, "
            f"total={ready_at - transition_started:.3f}s",
            flush=True,
        )

    def recover_from_step_error(err):
        nonlocal recording, next_t
        print(f"[sonic] simulator step error; re-sampling and continuing: {err}", flush=True)
        if recording:
            sync_exporter("x", from_vr=False)
            if env.ep_directory is not None:
                print(f"[sonic] discarded {os.path.basename(env.ep_directory)}", flush=True)
            env.has_interaction, env.states, env.integration_states, env.action_infos = False, [], [], []
            env.ep_directory = None
            recording = False
        reset_with_retry(wall, base, args)
        source.reset(base)
        announce_episode()
        next_t = time.perf_counter()

    try:
        while True:
            if instruction_pub is not None:
                instruction_pub.maybe_publish()
            p = keys.consume()
            vr_events = manager_sub.poll() if manager_sub is not None else set()
            if "b" in p:
                _controller(base).toggle_band()
            episode_event, from_vr = _resolve_episode_event(p, vr_events, recording)

            if episode_event == "start":
                if gains is None:
                    print("[sonic] not engaged yet -- cannot record.", flush=True)
                    if from_vr and keyboard_pub is not None:
                        keyboard_pub.send("x")
                else:
                    if instruction_pub is not None:
                        instruction_pub.mark_start()
                        if args.vla_instruction_start_delay > 0:
                            time.sleep(args.vla_instruction_start_delay)
                    if not from_vr:
                        sync_exporter("c", from_vr=False, delay=False)
                    env.start_episode_from_current_state(ep_meta=current_ep_meta)
                    recording = True
                    print("[sonic] recording...", flush=True)
            elif episode_event == "save":
                sync_exporter("c", from_vr=from_vr)
                finish(discard=False)
                continue
            elif episode_event == "discard":
                sync_exporter("x", from_vr=from_vr)
                finish(discard=True)
                continue

            a = source.act(base)
            if a is not None and source.gains:
                _controller(base).set_command_gains(source.gains)
                if _valid_sonic_gains(source.gains):
                    gains = source.gains
            act = a if a is not None else hold
            try:
                (env if (recording and a is not None) else wall).step(act)
                if image_pub is not None:
                    image_pub.maybe_publish(base)
            except mujoco.FatalError as e:
                recover_from_step_error(e)
                continue

            next_t += base.control_timestep
            slp = next_t - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n[sonic] stopping.", flush=True)
        if recording:
            finish(discard=False)
    finally:
        keys.close()
        if manager_sub is not None:
            manager_sub.close()
        if keyboard_pub is not None:
            keyboard_pub.close()
        if instruction_pub is not None:
            instruction_pub.close()
        if image_pub is not None:
            image_pub.close()
        if saved:
            h = gather_demonstrations_as_hdf5(eps, demo_dir, env_info, successful_episodes=saved,
                                              verbose=True)
            _stamp_sonic_attrs(h, saved_gains or gains, args)
            if h:
                convert_to_robomimic_format(h, filter_num_demos=None)
                print(f"[sonic] wrote {len(saved)} episode(s) -> {h}", flush=True)
        else:
            print("[sonic] no episodes saved.", flush=True)
        env.close()


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--environment", default="Kitchen", help="robocasa kitchen env name")
    ap.add_argument(
        "--layout",
        type=int,
        default=-1,
        help="kitchen layout id (-1=random compatible test layout)",
    )
    ap.add_argument("--style", type=int, default=None, help="kitchen style id (None=random)")
    ap.add_argument("--robot", default="SonicG1", help="SonicG1 or SonicG1Fixed")
    ap.add_argument(
        "--render-camera",
        default="robot0_frontview",
        help=(
            "MuJoCo camera used by the interactive viewer; "
            "use robot0_head_camera for robot first-person view"
        ),
    )
    ap.add_argument("--out", default="/tmp/sonic_robocasa_demos", help="output dataset directory")
    ap.add_argument("--record-freq", type=int, default=1, help="record one sample every N steps")
    ap.add_argument("--sim-dt", type=float, default=0.005, help="physics timestep (s); 200 Hz")
    ap.add_argument("--control-freq", type=int, default=200,
                    help="step() rate (Hz); = 1/sim-dt for one PD command per step")
    ap.add_argument("--post-action-hz", type=float, default=20.0,
                    help="rate (Hz) for the heavy per-step bookkeeping (reward/success/update_state)")
    ap.add_argument("--render-hz", type=float, default=20.0, help="viewer refresh rate (Hz)")
    ap.add_argument("--floor-friction", type=float, default=1.0, help="floor tangential friction")
    ap.add_argument("--floor-torsion", type=float, default=0.005, help="floor torsional friction")
    ap.add_argument("--wall-alpha", type=float, default=0.0,
                    help="enclosing-wall transparency in the viewer (0 hides, 1 opaque)")
    ap.add_argument("--rtf-log", action="store_true", help="print the [real-time] RTF line")
    ap.add_argument("--vla-stream", action="store_true",
                    help="Publish RoboCasa images and sync episode controls for run_data_exporter.py")
    ap.add_argument("--vla-camera-port", type=int, default=5555,
                    help="ZMQ port for the RoboCasa VLA camera stream")
    ap.add_argument("--vla-camera-name", default="robot0_head_camera",
                    help="MuJoCo camera rendered as the single VLA image stream")
    ap.add_argument("--vla-camera-key", default="ego_view",
                    help="Image key expected by run_data_exporter.py for single-camera streaming")
    ap.add_argument("--vla-camera-names", nargs="+", default=None,
                    help="MuJoCo cameras rendered into one VLA camera message")
    ap.add_argument("--vla-camera-keys", nargs="+", default=None,
                    help="Image keys for --vla-camera-names, e.g. ego_view left_wrist right_wrist")
    ap.add_argument("--vla-camera-width", type=int, default=640)
    ap.add_argument("--vla-camera-height", type=int, default=480)
    ap.add_argument("--vla-camera-hz", type=float, default=30.0,
                    help="Camera publish rate; 30 Hz matches the supported real OAK camera path")
    ap.add_argument("--vla-camera-flip", dest="vla_camera_flip", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--no-vla-camera-flip", dest="vla_camera_flip", action="store_false",
                    help="Do not vertically flip rendered frames before publishing")
    ap.add_argument("--vla-manager-host", default="localhost",
                    help="PICO manager_state ZMQ host")
    ap.add_argument("--vla-manager-port", type=int, default=5556,
                    help="PICO manager_state ZMQ port")
    ap.add_argument("--vla-keyboard-port", type=int, default=5580,
                    help="run_data_exporter.py ZMQ keyboard port")
    ap.add_argument("--no-vla-keyboard-sync", dest="vla_keyboard_sync", action="store_false",
                    help="Do not forward local c/k/x hotkeys to run_data_exporter.py")
    ap.add_argument("--vla-instruction-port", type=int, default=5581,
                    help="ZMQ port for per-episode RoboCasa instruction metadata")
    ap.add_argument(
        "--no-vla-instruction-sync",
        dest="vla_instruction_sync",
        action="store_false",
        help="Do not publish sampled episode instructions to run_data_exporter.py",
    )
    ap.add_argument("--vla-instruction-heartbeat", type=float, default=0.5,
                    help="Seconds between repeats of the current episode instruction")
    ap.add_argument("--vla-instruction-start-delay", type=float, default=0.05,
                    help="Local-start grace period for exporter instruction receipt (seconds)")
    ap.set_defaults(
        vla_keyboard_sync=True,
        vla_instruction_sync=True,
        vla_camera_flip=False,
    )
    ap.add_argument("--vla-save-sync-delay", type=float, default=0.08,
                    help="Small episode-end delay so run_data_exporter.py sees save/discard before reset")
    args = ap.parse_args()
    if (args.vla_camera_names is None) != (args.vla_camera_keys is None):
        ap.error("--vla-camera-names and --vla-camera-keys must be provided together")
    if args.vla_camera_names is None:
        args.vla_camera_names = [args.vla_camera_name]
        args.vla_camera_keys = [args.vla_camera_key]
    if len(args.vla_camera_names) != len(args.vla_camera_keys):
        ap.error("--vla-camera-names and --vla-camera-keys must have the same length")
    if len(set(args.vla_camera_keys)) != len(args.vla_camera_keys):
        ap.error("--vla-camera-keys must be unique")
    if args.vla_instruction_heartbeat <= 0:
        ap.error("--vla-instruction-heartbeat must be positive")
    if args.vla_instruction_start_delay < 0:
        ap.error("--vla-instruction-start-delay must be non-negative")
    return args


def main():
    args = get_args()
    if args.rtf_log:
        macros.CONSOLE_LOGGING_LEVEL = "DEBUG"
    macros.SIMULATION_TIMESTEP = args.sim_dt  # robosuite reads this in initialize_time

    cfg = load_composite_controller_config(controller=SONIC_CFG)
    base, env_kwargs = make_env(args, cfg)
    wall = EnclosingWallRenderWrapper(base, alpha=args.wall_alpha, enabled=True)
    reset_with_retry(wall, base, args)
    n_sub = int(round(base.control_timestep / base.model_timestep))
    solver = mujoco.mjtSolver(base.sim.model._model.opt.solver).name.removeprefix("mjSOL_")
    print(f"[sonic] {1.0/args.sim_dt:.0f} Hz physics | control_freq {args.control_freq} | "
          f"{n_sub} substep(s)/step | solver {solver}", flush=True)

    source = DDSActionSource(_controller(base)._cfg)
    source.reset(base)
    run_collection(args, base, wall, env_kwargs, source)


if __name__ == "__main__":
    main()
