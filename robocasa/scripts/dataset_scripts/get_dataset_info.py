"""
Helper script to report dataset information. By default, will print trajectory length statistics,
the maximum and minimum action element in the dataset, environment
metadata, and the structure of the first demonstration. If --verbose is passed, it will
report the structure of all demonstrations (not just the first one).

Args:
    dataset (str): path to hdf5 dataset

    verbose (bool): if flag is provided, print more details, like the structure of all
        demonstrations (not just the first one)

Example usage:

    # run script on example hdf5 packaged with repository
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5
"""
import json
import argparse
import numpy as np
from collections import OrderedDict
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import robocasa.utils.lerobot_utils as LU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output",
    )
    parser.add_argument(
        "--all_stats",
        action="store_true",
        help="get all detailed dataset statistics",
    )
    args = parser.parse_args()

    # extract demonstration list from file
    ds = LeRobotDataset(repo_id="robocasa365", root=args.dataset)

    episodes = ds.meta.episodes

    # extract length of each trajectory in the file
    traj_lengths = [episodes[ep]["length"] for ep in episodes]
    stats = ds.meta.stats
    action_min = stats["action"]["min"]
    action_max = stats["action"]["max"]
    traj_lengths = np.array(traj_lengths)

    env_meta = LU.get_env_metadata(ds.root)
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    for ep in episodes:
        print("episode {} with {} samples".format(ep, episodes[ep]["length"]))
        start_idx = int(ds.episode_data_index["from"][ep])

        ep_sample_0 = ds[start_idx]
        for k in ep_sample_0:
            if hasattr(ep_sample_0[k], "size"):
                print("    {}: shape {}".format(k, ep_sample_0[k].size()))
            else:
                print("    {}: {}".format(k, ep_sample_0[k]))

        if not args.verbose:
            break

    if args.all_stats:
        obj_cat_counts = {}
        layout_counts = {}
        style_counts = {}
        langs = []
        for ep in episodes:
            ep_meta = LU.get_episode_meta(ds.root, ep)
            langs.append(ep_meta["lang"])
            obj_cfgs = ep_meta["object_cfgs"]
            cat = None
            for cfg in obj_cfgs:
                if cfg["name"] == "obj":
                    cat = cfg["info"]["cat"]
                    break
            if cat not in obj_cat_counts:
                obj_cat_counts[cat] = 0
            obj_cat_counts[cat] += 1

            layout_id = ep_meta["layout_id"]
            style_id = ep_meta["style_id"]
            if layout_id not in layout_counts:
                layout_counts[layout_id] = 0
            if style_id not in style_counts:
                style_counts[style_id] = 0
            layout_counts[layout_id] += 1
            style_counts[style_id] += 1

        print()
        print("obj cat counts:", json.dumps(obj_cat_counts, indent=4))
        print(
            "layout_counts:",
            json.dumps(OrderedDict(sorted(layout_counts.items())), indent=4),
        )
        print(
            "style_counts:",
            json.dumps(OrderedDict(sorted(style_counts.items())), indent=4),
        )
        print("num unique lang instructions:", len(set(langs)))

    print("")
    print("total samples: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))

    # maybe display error message
    print("")
    if (action_min < -1.0).any() or (action_max > 1.0).any():
        raise Exception(
            "Dataset should have actions in [-1., 1.] but got bounds [{}, {}]".format(
                action_min, action_max
            )
        )
