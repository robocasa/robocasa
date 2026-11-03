import argparse
import json
import os
import random
import time
import h5py
import mujoco
import imageio
import numpy as np
from termcolor import colored
import traceback

import robosuite
import robosuite.macros as macros
import robocasa

from robocasa.scripts.dataset_scripts.playback_utils import (
    resolve_instruction_from_ep_meta,
)


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
    sonic_gains=None,
    sonic_runtime=None,
    integration_states=None,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    action_playback = actions is not None

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = resolve_instruction_from_ep_meta(ep_meta)
        if lang:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)
    _apply_sonic_runtime(
        env,
        initial_state,
        sonic_gains,
        sonic_runtime,
        integration_states,
        require_action_replay_metadata=action_playback,
    )

    traj_len = states.shape[0]
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    if render is False:
        print(colored("Running episode...", "yellow"))

    for t in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[t])
            if t < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[t + 1], state_playback)):
                    err = np.linalg.norm(states[t + 1] - state_playback)
                    if verbose or t == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, t
                                ),
                                "yellow",
                            )
                        )
        else:
            reset_to(env, {"states": states[t]})

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if t % video_skip == 0 or t == traj_len - 1:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally

                video_writer.append_data(video_img)

            # video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def _is_sonic_env(env):
    try:
        return type(env.robots[0].composite_controller).__name__ == "SonicWholeBodyController"
    except Exception:
        return False


def _decode_json_attr(value):
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value) if value else {}


def _env_meta_uses_sonic(env_meta):
    try:
        robots = env_meta.get("env_kwargs", {}).get("robots", [])
        if isinstance(robots, str):
            robots = [robots]
        return any(str(robot).startswith("SonicG1") for robot in robots)
    except Exception:
        return False


def _load_sonic_gains(env, sonic_gains_json, require=False, disable_band=False):
    """Load recorded SONIC PD gains for action replay. The dataset's stamped sonic_gains are
    authoritative; do not repair missing or invalid gains from the controller config."""
    if not _is_sonic_env(env):
        return
    gains = _decode_json_attr(sonic_gains_json)
    body_kp = np.asarray(gains["body"][0], dtype=float) if gains.get("body") else None
    if not gains or body_kp is None or not np.any(np.abs(body_kp) > 1e-9):
        if require:
            raise ValueError(
                "SONIC action replay requires valid dataset-level sonic_gains "
                "with nonzero body kp; refusing to fall back to controller config."
            )
        return
    env.robots[0].composite_controller.set_command_gains({
        k: (np.asarray(kp, dtype=float), np.asarray(kd, dtype=float))
        for k, (kp, kd) in gains.items()
    })
    if disable_band:
        env.robots[0].composite_controller.release_band()


def _restore_sonic_integration_state(env, integration_states):
    """Restore optional mjSTATE_INTEGRATION captured by newer SONIC demos. The critical field
    for contact-sensitive SONIC walking replay is qacc_warmstart; restoring the full first
    integration state also preserves the exact initial MuJoCo state vector."""
    if integration_states is None or not _is_sonic_env(env):
        return
    model = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.asarray(integration_states[0], dtype=float)
    expected = mujoco.mj_stateSize(model, spec)
    if state.size != expected:
        raise ValueError(
            f"SONIC states_integration has size {state.size}, expected {expected}."
        )
    mujoco.mj_setState(model, data, state, spec)
    env.sim.forward()


def _refresh_sonic_part_controller_state(env):
    """Refresh cached joint pos/vel after playback loads a saved simulator state."""
    if not _is_sonic_env(env):
        return
    for part_ctrl in env.robots[0].composite_controller.part_controllers.values():
        part_ctrl.update(force=True)


def _apply_sonic_runtime(
    env,
    initial_state,
    sonic_gains_json,
    sonic_runtime_json,
    integration_states=None,
    require_action_replay_metadata=False,
):
    """Restore SONIC runtime settings that are not preserved in robomimic env_args/XML."""
    if not _is_sonic_env(env):
        return
    runtime = _decode_json_attr(sonic_runtime_json)
    try:
        from robosuite.scripts.collect_sonic_g1_demos import match_base_sim_physics
        match_base_sim_physics(
            env.sim.model._model,
            floor_friction=float(runtime.get("floor_friction", 1.0)),
            floor_torsion=float(runtime.get("floor_torsion", 0.005)),
            timestep=float(runtime.get("sim_dt", env.control_timestep)),
        )
        env.sim.forward()
    except Exception:
        pass

    # Collection throttles robocasa's expensive fixture update_state loop. Replaying with the
    # default every-step post_action changes fixture dynamics and can seed contact divergence.
    post_freq = runtime.get("post_action_freq")
    if post_freq is None:
        post_freq = max(1, round(float(getattr(env, "control_freq", 20)) / 20.0))
    env.post_action_freq = int(post_freq)
    env.render_freq = int(runtime.get("render_freq", env.post_action_freq))
    if "states" in initial_state:
        try:
            env.timestep = int(round(float(initial_state["states"][0]) / env.control_timestep))
            env.cur_time = float(initial_state["states"][0])
        except Exception:
            pass

    _load_sonic_gains(
        env,
        sonic_gains_json,
        require=require_action_replay_metadata,
        disable_band=require_action_replay_metadata,
    )
    _restore_sonic_integration_state(env, integration_states)
    if require_action_replay_metadata:
        _refresh_sonic_part_controller_state(env)


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def _sonic_dataset_timestep(dataset):
    try:
        with h5py.File(dataset, "r") as f:
            runtime = _decode_json_attr(f["data"].attrs.get("sonic_runtime"))
            if "sim_dt" in runtime:
                return float(runtime["sim_dt"])
            env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
            env_kwargs = env_args.get("env_kwargs", {})
            robots = env_kwargs.get("robots", [])
            if any(str(r).startswith("SonicG1") for r in robots):
                cf = float(env_kwargs.get("control_freq", 200))
                return 1.0 / cf
    except Exception:
        pass
    return None


def playback_dataset(
    dataset,
    use_actions,
    use_abs_actions,
    use_obs,
    filter_key,
    n,
    render,
    render_image_names,
    camera_height,
    camera_width,
    video_path,
    video_skip,
    extend_states,
    first,
    verbose,
):
    # some arg checking
    write_video = render is not True
    if video_path is None:
        video_path = dataset.split(".hdf5")[0] + ".mp4"
        if use_actions:
            video_path = dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif use_abs_actions:
            video_path = dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
        render_image_names = "robot0_agentview_center"

    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    if use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not use_actions and not use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    env = None
    sonic_dataset = False

    # create environment only if not playing back with observations
    if not use_obs:
        # # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # # for playback since observations are unused. Pass a dummy spec here.
        # dummy_spec = dict(
        #     obs=dict(
        #             low_dim=["robot0_eef_pos"],
        #             rgb=[],
        #         ),
        # )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
        sonic_dataset = _env_meta_uses_sonic(env_meta)
        if use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = False

        if verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )

        env = robosuite.make(**env_kwargs)

    f = h5py.File(dataset, "r")
    sonic_gains_json = f["data"].attrs.get("sonic_gains")
    sonic_runtime_json = f["data"].attrs.get("sonic_runtime")

    # list of all demonstration episodes (sorted in increasing number order)
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = [
            elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if n is not None:
        random.shuffle(demos)
        demos = demos[:n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=video_skip,
                image_names=render_image_names,
                first=first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            use_actions and use_abs_actions
        )  # cannot use both relative and absolute actions
        if use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions
        integration_states = f["data/{}".format(ep)]["states_integration"][()] if "states_integration" in f["data/{}".format(ep)] else None
        if actions is not None and integration_states is None and (sonic_dataset or _is_sonic_env(env)):
            raise ValueError(
                "SONIC action replay requires states_integration. This dataset only has "
                "flattened MuJoCo states, so the initial warmstart/integration state is missing."
            )

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=render,
            video_writer=video_writer,
            video_skip=video_skip,
            camera_names=render_image_names,
            first=first,
            verbose=verbose,
            camera_height=camera_height,
            camera_width=camera_width,
            sonic_gains=sonic_gains_json,
            sonic_runtime=sonic_runtime_json,
            integration_states=integration_states,
        )

    f.close()
    if write_video:
        print(colored(f"Saved video to {video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()
    del env
    del video_writer


def get_playback_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_head_camera",
            "robot0_left_wrist_camera",
            "robot0_right_wrist_camera",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=768,
        help="(optional, for offscreen rendering) width of image observations",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    dataset_list = []
    if os.path.isdir(args.dataset):
        for root, dirs, files in os.walk(args.dataset):
            for file in files:
                if file == "demo.hdf5":
                    # with open(os.path.join(root, "ep_stats.json"), "r") as stats_f:
                    #     ep_stats = json.load(stats_f)
                    # stale = ep_stats.get("stale", False)
                    # if stale:
                    #     continue
                    if os.path.exists(os.path.join(root, "demo.mp4")):
                        # already recorded video
                        continue
                    dataset_list.append(os.path.join(root, file))
    else:
        dataset_list = [args.dataset]

    dataset_exceptions = []
    for ds_i, dataset in enumerate(dataset_list):
        print(
            colored(
                f"\n[{ds_i+1}/{len(dataset_list)}] Playing back {dataset}", "yellow"
            )
        )
        try:
            sonic_dt = _sonic_dataset_timestep(dataset)
            if sonic_dt is not None:
                macros.SIMULATION_TIMESTEP = sonic_dt
            playback_dataset(
                dataset=dataset,
                use_actions=args.use_actions,
                use_abs_actions=args.use_abs_actions,
                use_obs=args.use_obs,
                filter_key=args.filter_key,
                n=args.n,
                render=args.render,
                render_image_names=args.render_image_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                video_path=args.video_path,
                video_skip=args.video_skip,
                extend_states=args.extend_states,
                first=args.first,
                verbose=args.verbose,
            )
        except KeyboardInterrupt:
            print(colored(f"Exiting Playback Early.", "yellow"))
            break
        except Exception as e:
            stack_trace = traceback.format_exc()
            print(colored("Exception!", "red"))
            print(colored(f"{stack_trace}", "red"))
            dataset_exceptions.append((dataset, stack_trace))
            print(
                colored(
                    f"[{len(dataset_exceptions)}/{ds_i+1}] exceptions so far.\n", "red"
                )
            )

    if len(dataset_exceptions) > 0:
        print()
        print(
            colored(f"Playback failed with the following resulting in errors:", "red")
        )
        for (dataset, stack_trace) in dataset_exceptions:
            print(colored(f"{dataset}:", "red"))
            # print(colored(f"{stack_trace}\n", "red"))
        raise SystemExit(1)
