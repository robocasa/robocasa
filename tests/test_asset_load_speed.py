import os
import numpy as np
import tqdm
import traceback
import argparse

from robocasa.scripts.browse_mjcf_model import read_model
import robocasa


def find_all_xml_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                file_list.append(os.path.join(root, file))
    return file_list


def get_load_time_stats(directory, verbose=False):
    xml_files = find_all_xml_files(directory)
    load_times = []

    for mjcf_path in xml_files:
        try:
            sim, info = read_model(filepath=mjcf_path)
            load_times.append(info["sim_load_time"])
            del sim
        except Exception as e:
            if verbose:
                print("Caught exception for {}".format(mjcf_path))
                traceback.print_exc()

    if len(load_times) == 0:
        return None

    stats = dict(
        mean=np.mean(load_times),
        min=np.min(load_times),
        percentile25=np.percentile(load_times, 25),
        percentile50=np.percentile(load_times, 50),
        percentile75=np.percentile(load_times, 75),
        max=np.max(load_times),
    )

    print("Directory:", directory)
    print("Mean loading time: {:.4f} s".format(stats["mean"]))
    print("Median loading time: {:.4f} s".format(stats["percentile50"]))
    print()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()

    if args.directory is not None:
        get_load_time_stats(args.directory)
    else:
        objects_path = os.path.join(robocasa.__path__[0], "models/assets/objects")
        fixtures_path = os.path.join(robocasa.__path__[0], "models/assets/fixtures")
        objaverse_path = os.path.join(objects_path, "objaverse")

        for fixture_type in os.listdir(fixtures_path):
            dir = os.path.join(fixtures_path, fixture_type)
            get_load_time_stats(dir)

        get_load_time_stats(objaverse_path)
