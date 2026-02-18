import argparse
import json
import os
import random
import time
import h5py
import imageio
import numpy as np
from termcolor import colored
import traceback

import robosuite
import robocasa
from pathlib import Path
import robocasa.utils.lerobot_utils as LU


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

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert (
            states.shape[0] == actions.shape[0]
        ), "number of states and actions must match!, got {} vs {}".format(
            states.shape[0], actions.shape[0]
        )

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
    dataset = Path(dataset)
    # some arg checking
    write_video = render is not True
    if video_path is None:
        video_path = str(dataset.parent) + ".mp4"
        if use_actions:
            video_path = str(dataset.parent) + "_use_actions.mp4"
        elif use_abs_actions:
            video_path = str(dataset.parent) + "_use_abs_actions.mp4"
    assert not (render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if render_image_names is None:
        # We fill in the automatic values
        env_meta = LU.get_env_metadata(dataset)
        render_image_names = ["robot0_agentview_center"]

    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    assert not use_obs, "Not supported with lerobot dataset format currently"

    env = None

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

        env_meta = LU.get_env_metadata(dataset)
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

    assert filter_key is None, "filter_key not supported for lerobot dataset format"
    demos = LU.get_episodes(dataset)

    # maybe reduce the number of demonstrations to playback
    if n is not None:
        demos = demos[:n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep.stem), "yellow"))

        # prepare initial state to reload from
        states = LU.get_episode_states(dataset, ind)
        initial_state = dict(states=states[0])
        initial_state["model"] = LU.get_episode_model_xml(dataset, ind)
        initial_state["ep_meta"] = json.dumps(LU.get_episode_meta(dataset, ind))

        if extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            use_actions and use_abs_actions
        )  # cannot use both relative and absolute actions
        if use_actions:
            actions = LU.get_episode_actions(dataset, ind, abs_actions=use_abs_actions)

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
        )

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
        help="path to lerobot dataset",
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
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
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
    dataset_list = [args.dataset]

    dataset_exceptions = []
    for ds_i, dataset in enumerate(dataset_list):
        print(
            colored(
                f"\n[{ds_i+1}/{len(dataset_list)}] Playing back {dataset}", "yellow"
            )
        )
        try:
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
