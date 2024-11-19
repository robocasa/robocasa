import os
import h5py
import json
import torch
import numpy as np

import robocasa.utils.robomimic.robomimic_torch_utils as TorchUtils


def get_env_metadata_from_dataset(dataset_path):
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
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta


def extract_action_dict(dataset):
    # find files
    f = h5py.File(os.path.expanduser(dataset), mode="r+")

    SPECS = [
        dict(
            key="actions",
            is_absolute=False,
        ),
        dict(
            key="actions_abs",
            is_absolute=True,
        ),
    ]

    # execute
    for spec in SPECS:
        input_action_key = spec["key"]
        is_absolute = spec["is_absolute"]

        if is_absolute:
            prefix = "abs_"
        else:
            prefix = "rel_"

        for demo in f["data"].values():
            if str(input_action_key) not in demo.keys():
                continue
            in_action = demo[str(input_action_key)][:]
            in_pos = in_action[:, :3].astype(np.float32)
            in_rot = in_action[:, 3:6].astype(np.float32)
            in_grip = in_action[:, 6:7].astype(np.float32)

            rot_6d = TorchUtils.axis_angle_to_rot_6d(
                axis_angle=torch.from_numpy(in_rot)
            )
            rot_6d = rot_6d.numpy().astype(np.float32)  # convert to numpy

            this_action_dict = {
                prefix + "pos": in_pos,
                prefix + "rot_axis_angle": in_rot,
                prefix + "rot_6d": rot_6d,
                "gripper": in_grip,
            }

            # special case: 8 dim actions mean there is a mobile base mode in the action space
            if in_action.shape[1] == 8:
                this_action_dict["base_mode"] = in_action[:, 7:8].astype(np.float32)

            action_dict_group = demo.require_group("action_dict")
            for key, data in this_action_dict.items():
                if key in action_dict_group:
                    del action_dict_group[key]
                action_dict_group.create_dataset(key, data=data)

    f.close()


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Creates a new hdf5 filter key in hdf5 file @hdf5_path with
    name @key_name that corresponds to the demonstrations
    @demo_keys. Filter keys are generally useful to create
    named subsets of the demonstrations in an hdf5, making it
    easy to train, test, or report statistics on a subset of
    the trajectories in a file.

    Returns the list of episode lengths that correspond to the filtering.

    Args:
        hdf5_path (str): path to hdf5 file
        demo_keys ([str]): list of demonstration keys which should
            correspond to this filter key. For example, ["demo_0",
            "demo_1"].
        key_name (str): name of filter key to create

    Returns:
        ep_lengths ([int]): list of episode lengths that corresponds to
            each demonstration in the new filter key
    """
    f = h5py.File(hdf5_path, "a")
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype="S")

    f.close()
    return ep_lengths


def filter_dataset_size(
    hdf5_path, num_demos, input_filter_key=None, output_filter_key=None
):
    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    if input_filter_key is not None:
        print("using filter key: {}".format(input_filter_key))
        demos = sorted(
            [
                elem.decode("utf-8")
                for elem in np.array(f["mask/{}".format(input_filter_key)])
            ]
        )
    else:
        demos = sorted(list(f["data"].keys()))
    f.close()

    # get random split
    total_num_demos = len(demos)
    mask = np.zeros(total_num_demos)
    mask[:num_demos] = 1.0
    np.random.shuffle(mask)
    mask = mask.astype(int)
    subset_inds = mask.nonzero()[0]
    subset_keys = [demos[i] for i in subset_inds]

    # pass mask to generate split
    if output_filter_key is not None:
        name = output_filter_key
    else:
        name = "{}_demos".format(num_demos)

    if input_filter_key is not None:
        name = "{}_{}".format(input_filter_key, name)

    subset_lengths = create_hdf5_filter_key(
        hdf5_path=hdf5_path, demo_keys=subset_keys, key_name=name
    )

    print("Total number of subset samples: {}".format(np.sum(subset_lengths)))
    print("Average number of subset samples {}".format(np.mean(subset_lengths)))


def move_demo_to_new_key(f, old_demo_key, new_demo_key, delete_old_demo=True):
    print(f"Moving {old_demo_key} -> {new_demo_key}")

    src_ep = f["data"][old_demo_key]

    if new_demo_key not in f["data"]:
        f["data"].create_group(new_demo_key)

    targ_ep = f["data"][new_demo_key]

    targ_ep.attrs.update(src_ep.attrs)

    # write the datasets
    for k in src_ep:
        if isinstance(src_ep[k], h5py.Dataset):
            targ_ep.create_dataset(k, data=np.array(src_ep[k]))
        else:
            for m in src_ep[k]:
                targ_ep.create_dataset(
                    "{}/{}".format(k, m),
                    data=np.array(src_ep["{}/{}".format(k, m)]),
                )

    # write the metadata present in attributes as well
    for k in src_ep.attrs:
        targ_ep.attrs[k] = src_ep.attrs[k]

    if delete_old_demo is True:
        del f["data"][old_demo_key]


def make_demo_ids_contiguous(dataset):
    f = h5py.File(dataset, "a")  # edit mode

    num_old_demos = max([int(demo_key.split("_")[-1]) for demo_key in f["data"]]) + 1
    missing_demo_inds = [
        i for i in range(num_old_demos) if f"demo_{i}" not in f["data"]
    ]
    num_new_demos = num_old_demos - len(missing_demo_inds)

    old_idx = num_old_demos - 1
    num_demos_changed = 0
    while num_demos_changed < len(missing_demo_inds):
        if f"demo_{old_idx}" not in f["data"]:
            old_idx -= 1
            continue

        new_idx = missing_demo_inds[num_demos_changed]

        if old_idx <= new_idx:
            break

        old_demo_key = f"demo_{old_idx}"
        new_demo_key = f"demo_{new_idx}"

        move_demo_to_new_key(f, old_demo_key, new_demo_key, delete_old_demo=True)

        old_idx -= 1
        num_demos_changed += 1

    f.close()
