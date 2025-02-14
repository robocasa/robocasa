"""
Helper script to report dataset information. By default, will print trajectory length statistics,
the maximum and minimum action element in the dataset, filter keys present, environment
metadata, and the structure of the first demonstration. If --verbose is passed, it will
report the exact demo keys under each filter key, and the structure of all demonstrations
(not just the first one).

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, report statistics on the subset of trajectories
        in the file that correspond to this filter key

    verbose (bool): if flag is provided, print more details, like the structure of all
        demonstrations (not just the first one)

Example usage:

    # run script on example hdf5 packaged with repository
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5

    # run script only on validation data
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5 --filter_key valid
"""
import h5py
import json
import argparse
import numpy as np

if __name__ == "__main__":
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
        help="(optional) if provided, report statistics on the subset of trajectories \
            in the file that correspond to this filter key",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output",
    )
    args = parser.parse_args()

    # extract demonstration list from file
    filter_key = args.filter_key
    all_filter_keys = None
    f = h5py.File(args.dataset, "r")
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted(
            [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        )
    else:
        # use all demonstrations
        demos = sorted(list(f["data"].keys()))

        # extract filter key information
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                )
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)

    # report statistics on the data
    print("")
    print("total transitions: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))
    print("")
    print("==== Filter Keys ====")
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print("filter key {} with {} demos".format(fk, len(all_filter_keys[fk])))
    else:
        print("no filter keys")
    print("")
    if args.verbose:
        if all_filter_keys is not None:
            print("==== Filter Key Contents ====")
            for fk in all_filter_keys:
                print(
                    "filter_key {} with {} demos: {}".format(
                        fk, len(all_filter_keys[fk]), all_filter_keys[fk]
                    )
                )
        print("")
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    for ep in demos:
        print(
            "episode {} with {} transitions".format(
                ep, f["data/{}".format(ep)].attrs["num_samples"]
            )
        )
        for k in f["data/{}".format(ep)]:
            if k in ["obs", "next_obs"]:
                print("    key: {}".format(k))
                for obs_k in f["data/{}/{}".format(ep, k)]:
                    shape = f["data/{}/{}/{}".format(ep, k, obs_k)].shape
                    print(
                        "        observation key {} with shape {}".format(obs_k, shape)
                    )
            elif isinstance(f["data/{}/{}".format(ep, k)], h5py.Dataset):
                key_shape = f["data/{}/{}".format(ep, k)].shape
                print("    key: {} with shape {}".format(k, key_shape))

        if not args.verbose:
            break

    obj_cat_counts = {}
    layout_counts = {}
    style_counts = {}
    langs = []
    for ep in demos:
        ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
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

    # for k, v in obj_cat_counts.items():
    #     print(k, v)
    print()
    print("obj cat counts:", json.dumps(obj_cat_counts, indent=4))
    print("layout_counts:", json.dumps(layout_counts, indent=4))
    print("style_counts:", json.dumps(style_counts, indent=4))
    print("num unique lang instructions:", len(set(langs)))

    f.close()

    # maybe display error message
    print("")
    if (action_min < -1.0) or (action_max > 1.0):
        raise Exception(
            "Dataset should have actions in [-1., 1.] but got bounds [{}, {}]".format(
                action_min, action_max
            )
        )
