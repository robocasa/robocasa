import argparse
import os

import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="path to base path containing hdf5 datasets",
    )
    args = parser.parse_args()

    hdf5_paths = []
    for root, dirs, files in os.walk(args.base_path):
        for name in files:
            if name.endswith(".hdf5"):
                hdf5_paths.append(os.path.join(root, name))

    for path in hdf5_paths:
        f = h5py.File(path, "a")
        for ep_key in f["data"].keys():
            ep_data_grp = f["data"][ep_key]

            actions = ep_data_grp["actions"][:]

            if actions.shape[1] == 12:
                continue
            actions = np.concatenate(
                (actions, -1 * np.ones((actions.shape[0], 1))), axis=1
            )
            del ep_data_grp["actions"]
            ep_data_grp.create_dataset("actions", data=np.array(actions))

        f.close()
