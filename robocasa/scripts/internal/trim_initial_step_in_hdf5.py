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

            for k in ep_data_grp:
                if isinstance(ep_data_grp[k], h5py.Dataset):
                    new_data = ep_data_grp[k][1:]
                    del ep_data_grp[k]
                    ep_data_grp.create_dataset(k, data=np.array(new_data))
                if isinstance(ep_data_grp[k], h5py.Group):
                    for sub_k in ep_data_grp[k]:
                        assert isinstance(ep_data_grp[k][sub_k], h5py.Dataset)
                        new_data = ep_data_grp[k][sub_k][1:]
                        del ep_data_grp[k][sub_k]
                        ep_data_grp[k].create_dataset(sub_k, data=np.array(new_data))

        f.close()
