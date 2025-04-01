import argparse
import h5py
import numpy as np

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--input_filter_key",
        type=str,
        default=None,
        help="if provided, split the subset of trajectories in the file that correspond to\
            this filter key into a training and validation set of trajectories, instead of\
            splitting the full set of trajectories",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        nargs="+",
        default=[
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            75,
            80,
            90,
            100,
            125,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1500,
            2000,
            2500,
        ]
        + [n * 1000 for n in range(3, 21)],
    )
    parser.add_argument(
        "--output_filter_key",
        type=str,
        required=False,
        help="(optional) use custom name for output filter key name",
    )
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    for n in args.num_demos:
        DatasetUtils.filter_dataset_size(
            args.dataset,
            input_filter_key=args.input_filter_key,
            num_demos=n,
            output_filter_key=args.output_filter_key,
        )
