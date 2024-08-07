import argparse
import json
import os
import random
from pathlib import Path

import h5py
import imageio
import mujoco
import numpy as np
import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase, EnvType
from tqdm import tqdm

from robocasa.utils.usd.exporter import USDExporter

front_camera_pos = {
    0: (2.25, -5.74, 1.75),
    1: (2.65, -6.17, 1.9),
    2: (0.03418, -5.66508, 2.46522),
    3: (6.415, -5.16, 2.58),
    4: (1.6, -7, 1.749),
    5: (2.077, -6.369, 1.922),
    6: (3.2, -7.56, 2.3),
    7: (2.2, -7.613, 1.91),
    8: (2.248, -7.643, 1.81),
    9: (4.212, -6.63, 2),
}

front_camera_angle = {
    0: (84.09, 0, 0),
    1: (80.24, 0, 0),
    2: (72.10401, 0, -41.11511),
    3: (73.23, 0, 58.19),
    4: (79, 0, 0),
    5: (79.755, 0, 0.103),
    6: (76.93, 0, 0.786),
    7: (78.139, 0, 0),
    8: (79.98, 0, 0),
    9: (76.754, 0, 0),
}

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    demo_name,
    env,
    initial_state,
    states,
    save_dir,
    actions=None,
    camera_names=None,
    first=False,
    ep_name="tmp",
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
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """

    print(f"playing demonstration {demo_name}...")

    ep_meta = json.loads(initial_state["ep_meta"])
    layout_id = int(ep_meta["layout_id"])

    assert isinstance(env, EnvBase)

    env.reset_to(initial_state)

    model = env.env.sim.model._model
    data = env.env.sim.data._data

    renderer = USDExporter(
        model,
        light_intensity=100000,
        camera_names=[
            "robot0_eye_in_hand",
            "robot0_agentview_left",
            "robot0_agentview_right",
        ],
        output_directory_name=f"{ep_name}",
        output_directory_root=save_dir,
        specialized_materials_file="C:/Users/abhishek/Documents/research/robomimic-kitchen/robomimic/scripts/omniverse_materials.json",
    )

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in tqdm(range(traj_len)):

        env.reset_to({"states": states[i]})

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup = [0, 1, 1, 0, 0, 0]

        renderer.update_scene(data, scene_option=scene_option)

    renderer.add_camera(
        list(front_camera_pos[layout_id]), list(front_camera_angle[layout_id]), objid=1
    )

    renderer.add_light(
        pos=[0.0, 0.0, 0.0], intensity=4000, objid="dome_light", light_type="dome"
    )

    renderer.save_scene(filetype="usd")


def playback_dataset(dataset, args):
    save_dir = dataset.split(".hdf5")[0] + "_usd"

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    # need to make sure ObsUtils knows which observations are images, but it doesn't matter
    # for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset)
    # env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False # absolute action space
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False
    )

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.demo_key is not None:
        demos = [demo_key]

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            demo_name=Path(dataset).stem,
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            camera_names=args.render_image_names,
            first=args.first,
            save_dir=save_dir,
            ep_name=ep,
        )

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="the hdf5 dataset")

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    parser.add_argument(
        "--demo_key", type=int, default=None, help="(optional) a single demo to convert"
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

    args = parser.parse_args()

    playback_dataset(args.dataset, args)
