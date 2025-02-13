from robocasa.scripts.playback_dataset import (
    playback_trajectory_with_env,
    get_env_metadata_from_dataset,
    reset_to,
)
from robocasa.utils.qa_utils import (
    create_env_from_env_meta,
    scan_datset_quality,
    playback_demos,
    QA_MATRIC_TO_COLOR,
)

import argparse
from termcolor import colored
import os
import json
import traceback


def auto_inspect_ep(ds_path, env=None):
    delete_env_at_end = False
    if env is None:
        env_meta = get_env_metadata_from_dataset(dataset_path=ds_path)
        env = create_env_from_env_meta(env_meta)
        delete_env_at_end = True

    try:
        all_stats = scan_datset_quality(ds_path, env=env)
    except Exception as e:
        # skip if there's an issue
        print("Exception!")
        print(traceback.print_exc())
        return

    summary_stats = dict()
    for ep in all_stats:
        for metric in all_stats[ep]:
            if metric not in summary_stats:
                summary_stats[metric] = []

            flagged_timesteps = all_stats[ep][metric]
            # if at least one timestep is flagged, add this episode to list of flagged episodes for this metric
            if len(flagged_timesteps) > 0:
                summary_stats[metric].append(ep)

    qa_path = os.path.dirname(ds_path)
    os.makedirs(qa_path, exist_ok=True)

    all_flagged_eps = set([ep for ep_list in summary_stats.values() for ep in ep_list])

    print(
        "Total number of bad demos:",
        len(all_flagged_eps),
    )
    for metric in summary_stats:
        print(f"Num flagged demos due to {metric}: {len(summary_stats[metric])}")

    print("\nPlaying back demos...")
    video_path = os.path.join(qa_path, f"qa.mp4")
    demo_keys = summary_stats[metric]  # [: args.num_demos_to_render]

    highlight_frames_list = []
    demo_keys = list(all_stats.keys())
    for ep in demo_keys:
        highlight_frames = []
        for metric in all_stats[ep]:
            timesteps = all_stats[ep][metric]
            color = QA_MATRIC_TO_COLOR[metric]
            highlight_frames += zip(
                timesteps, [color] * len(timesteps), [metric] * len(timesteps)
            )
        highlight_frames_list.append(highlight_frames)
    playback_demos(
        ds_path,
        demo_keys,
        highlight_frames_list=highlight_frames_list,
        video_path=video_path,
        env=env,
    )

    qa_stats_path = os.path.join(qa_path, "qa_stats.json")
    if os.path.exists(qa_stats_path):
        with open(qa_stats_path, "r") as f:
            qa_stats = json.load(f)
    else:
        qa_stats = dict()

    qa_stats["auto_qa"] = dict(
        summary_stats=summary_stats,
        all_stats=all_stats,
    )
    with open(qa_stats_path, "w") as f:
        f.write(json.dumps(qa_stats, indent=4))

    if delete_env_at_end:
        env.close()
        del env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--overwrite",
        help="re-scan existing demos (only valid if args.dataset is a directory)",
        action="store_true",
    )
    args = parser.parse_args()

    demo_group_list = []
    args.dataset = os.path.expanduser(args.dataset)
    if os.path.isfile(args.dataset):
        demo_group_list = [[args.dataset]]
    else:
        ep_dir_list = []
        for root, dirs, files in os.walk(args.dataset):
            for dir in dirs:
                if dir == "episodes":
                    ep_dir_list.append(os.path.join(root, dir))

        for ep_dir in ep_dir_list:
            demo_group = []
            for root, dirs, files in os.walk(ep_dir):
                for file in files:
                    if file == "ep_demo.hdf5":
                        # make sure we only add successful episodes
                        try:
                            with open(
                                os.path.join(root, "ep_stats.json"), "r"
                            ) as stats_f:
                                ep_stats = json.load(stats_f)

                            if not ep_stats["success"]:
                                continue

                            # if the qa stats already exist, skip
                            if (
                                not args.overwrite
                                and os.path.exists(os.path.join(root, "qa_stats.json"))
                                and os.path.exists(os.path.join(root, "qa.mp4"))
                            ):
                                continue
                        except:
                            continue

                        demo_group.append(os.path.join(root, file))

            if len(demo_group) > 0:
                demo_group_list.append(demo_group)

    total_num_datasets = sum([len(demo_group) for demo_group in demo_group_list])
    num_datasets_scanned = 0
    for demo_group in demo_group_list:
        env_meta = get_env_metadata_from_dataset(dataset_path=demo_group[0])
        env = create_env_from_env_meta(env_meta)

        for ds_path in demo_group:
            # scan dataset quality
            print(
                colored(
                    f"[{num_datasets_scanned+1}/{total_num_datasets}] Auto inspecting: {ds_path}",
                    "yellow",
                )
            )
            auto_inspect_ep(ds_path, env=env)
            print()
            num_datasets_scanned += 1

        if env is not None:
            env.close()
            del env
