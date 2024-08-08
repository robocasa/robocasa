import argparse
import json
import os

import h5py
import numpy as np

# TODO: add option to delete trajectories that fail threshold automatically/copy over trajectories that are valid


def search_for_value(arr, val, num_conseq, start_pos):
    counter = 0
    for i in range(start_pos, len(arr)):
        if arr[i] == val:
            counter += 1
            if counter == num_conseq:
                return i - num_conseq + 1
        else:
            counter = 0
    return -1


def extract_key_points(traj):
    actions = traj["actions"]
    gripper_actions = np.array([action[-2] for action in actions])
    # locate when the gripper first closed
    # (defined as first five consecutive time steps to keep gripper closed)
    start_grip = search_for_value(gripper_actions, 1, num_conseq=5, start_pos=0)
    # locate when the gripper is opened
    end_grip = search_for_value(gripper_actions, -1, num_conseq=5, start_pos=start_grip)
    # return normalized value
    return start_grip / len(actions) * 100, end_grip / len(actions) * 100


def extract_metadata(dataset_path, meta):
    for layout_id in os.listdir(dataset_path):
        if "." in layout_id:
            continue

        # locate dataset file
        layout_folder_path = os.path.join(dataset_path, layout_id)
        demos = [
            f
            for f in os.listdir(layout_folder_path)
            if os.path.isdir(os.path.join(layout_folder_path, f))
        ]
        print(
            "{} demos found for layout {}, using the first one.".format(
                len(demos), layout_id
            )
        )

        # load dataset
        dataset_path = os.path.join(layout_folder_path, demos[0], "demo.hdf5")
        print("Opening:", dataset_path)
        dataset = h5py.File(dataset_path, "r")
        data = dataset["data"]

        # extract and store keypoints
        for demo in data.keys():
            start_grip, end_grip = extract_key_points(data[demo])
            print("{}: {} - {}".format(demo, round(start_grip, 5), round(end_grip, 5)))
            meta["key_points"]["start_grip"].append(start_grip)
            meta["key_points"]["end_grip"].append(end_grip)
            meta["layout_id"].append(layout_id)
            meta["demo_name"].append(demo)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument(
        "--base_dataset_path",
        default="/Users/lancezhang/projects/kitchen/robosuite/robosuite/models/assets/demonstrations_private",
        help="folder in which all datasets are stored (i.e. path to `demonstrations_private`)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="maximum number of error (timesteps) for a trajectory to be considered valid",
    )
    args = parser.parse_args()

    task_dataset_path = os.path.join(args.base_dataset_path, args.dataset_name)
    print("Scanning task_dataset:", task_dataset_path, "\n")

    # extract necessary metadata from dataset
    metadata = {
        "key_points": {"start_grip": list(), "end_grip": list()},
        "layout_id": list(),
        "demo_name": list(),
    }
    extract_metadata(task_dataset_path, metadata)

    # calculate statistics
    print("=" * 100, "\n")
    print("Total demos found:", len(metadata["demo_name"]))

    # calculate disalignment
    # (defined by mean # of frames trajectories deviate from keypoints)
    disalignment = 0
    key_points = metadata["key_points"]
    invalid_trajectories = dict()

    for i, k in enumerate(key_points.keys()):
        key_point_mean = np.mean(key_points[k])
        print("key point {} average: {}".format(i, round(key_point_mean, 5)), end=", ")
        avg_abs_err = list()

        # check for invalid trajectories and store information
        for j, key_point in enumerate(key_points[k]):
            abs_err = np.abs(key_point - key_point_mean)
            if abs_err > args.threshold:
                invalid_name = "layout {}: {}".format(
                    metadata["layout_id"][j], metadata["demo_name"][j]
                )
                if invalid_name not in invalid_trajectories:
                    invalid_trajectories[invalid_name] = list()
                invalid_trajectories[invalid_name].append(
                    ("key_point_{}".format(i), abs_err)
                )
            avg_abs_err.append(abs_err)
        avg_abs_err = np.mean(avg_abs_err)

        print("average absolute error: {} frames".format(round(avg_abs_err, 5)))
        disalignment += avg_abs_err

    print("Overall disalignment:", round(disalignment, 5))
    print()

    if len(invalid_trajectories) > 0:
        print("Invalid demos and associated errors:")
        for k, v in invalid_trajectories.items():
            print(k, "--", v)
    else:
        print("No invalid trajectories detected!")
