import argparse
from termcolor import colored
import os
import json
import cv2
import h5py
import time
import re

from robocasa.scripts.qa.run_qa_auto import auto_inspect_ep

COMMON_FAULTS = [
    dict(
        codename="jerk",
        description="robot motion is too sudden or jerky",
    ),
    dict(
        codename="awk_robot_pose",
        description="arm gets into awkward pose",
    ),
    dict(
        codename="stall",
        description="robot moving too slowly or having long abnormal pauses",
    ),
    dict(
        codename="coll",
        description="undesirable collisions with environment",
    ),
    dict(
        codename="awk_manipulation",
        description="awkward maniplation of object, door, knob, lever, etc",
    ),
    dict(
        codename="bad_obj_interaction",
        description="object slip, drop, wobble, drag, knock, throw, etc",
    ),
    dict(
        codename="too_many_retries",
        description="robot unsuccessfully tries to manipulate an object too many times",
    ),
    dict(
        codename="awk_nav",
        description="awkward robot base navigation",
    ),
    dict(
        codename="unnec_base",
        description="unnecessary usage of base",
    ),
    dict(
        codename="misc",
        description="other issue",
    ),
]

FAULT_OPTIONS_MSG = ""
for i, fault in enumerate(COMMON_FAULTS):
    FAULT_OPTIONS_MSG += f"[{i}] {fault['codename']}: {fault['description']}\n"


def play_mp4_opencv(file_path, height=500):
    """
    code adapted from Google Gemini
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(
            frame, (height * 3, height), interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Video", resized_frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_multi_numeric_input(input_str, num_range):
    parsed_inputs = []
    if len(input_str) == 0:
        return None
    for ch in input_str:
        if not ch.isdigit():
            return None
        if int(ch) not in num_range:
            return None
        parsed_inputs.append(int(ch))
    return list(set(parsed_inputs))


def write_manual_qa(qa_stats_path, user_action, user_issues=None, user_notes=None):
    if os.path.exists(qa_stats_path):
        with open(qa_stats_path, "r") as f:
            all_qa_stats = json.load(f)
    else:
        all_qa_stats = dict()

    status = None
    if user_action == "y":
        status = "pass"
    elif user_action == "n":
        status = "fail"
    elif user_action == "s":
        status = "maybe"
    else:
        raise ValueError

    if user_issues is None:
        issues = None
    else:
        issues = dict()
        for idx in range(len(COMMON_FAULTS)):
            codename = COMMON_FAULTS[idx]["codename"]
            issues[codename] = True if idx in user_issues else False

    all_qa_stats["manual_qa"] = dict(
        status=status,
        issues=issues,
        notes=user_notes,
    )

    with open(qa_stats_path, "w") as f:
        json.dump(all_qa_stats, f, indent=4)


def manual_inspect_ep(ds_path, video_height=500):
    qa_video = os.path.join(os.path.dirname(ds_path), "qa.mp4")
    if os.path.exists(qa_video) is False:
        print(
            colored(
                f"Could not find existing QA video. Running Auto QA first...", "yellow"
            )
        )
        auto_inspect_ep(ds_path)
    with h5py.File(ds_path) as f:
        env_name = f["data"].attrs["env"]
        assert len(f["data"]) == 1, "episode must have a single demo only"
        ep_meta = json.loads(f["data/demo_1"].attrs["ep_meta"])
        instr = ep_meta["lang"]
    print(colored(f"\tTask: {env_name}", "yellow"))
    print(colored(f"\tInstruction: {instr}", "yellow"))
    time.sleep(2.5)
    while True:
        play_mp4_opencv(qa_video, height=video_height)

        while True:
            user_action = input("keep? yes(y) / no(n) / maybe(s) / replay(r) ")
            if user_action in ["y", "n", "s", "r"]:
                break
            else:
                print("Invalid input. Please try again.")

        if user_action == "r":
            print(colored("Replaying...", "yellow"))
            continue
        else:
            break

    if user_action == "y":
        print(colored("Episode marked as *keep*", "yellow"))
        user_issues = None
        user_notes = input("Any notes? (or press enter) ")
    elif user_action in ["n", "s"]:
        if user_action == "n":
            print(colored("Episode marked as *discarded*", "yellow"))
        elif user_action == "s":
            print(colored("Episode marked as *maybe keep*", "yellow"))
        print(FAULT_OPTIONS_MSG)
        while True:
            user_issues = input("Mark all issues: ")
            user_issues = parse_multi_numeric_input(
                user_issues, range(len(COMMON_FAULTS))
            )
            if user_issues is not None:
                break
            else:
                print("Invalid input. Please try again.")
        user_notes = input("Any notes? (or press enter) ")

    qa_stats_path = os.path.join(os.path.dirname(ds_path), "qa_stats.json")
    write_manual_qa(
        qa_stats_path,
        user_action,
        user_issues=user_issues,
        user_notes=user_notes,
    )
    print(colored("Saved QA to " + qa_stats_path, "green"))
    print()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--from_date",
        type=str,
        help="(optional) only scan folders from date yyyy-mm-dd",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=500,
        help="(optional) height of video played back on screen",
    )
    args = parser.parse_args()

    if args.from_date is not None:
        assert bool(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.from_date)
        ), "date format must be yyyy-mm-dd"

    ep_list = []
    args.dataset = os.path.expanduser(args.dataset)
    if os.path.isfile(args.dataset):
        ep_list = [args.dataset]
    else:
        ep_list = []
        for root, dirs, files in os.walk(args.dataset):
            for file in files:
                if file == "ep_demo.hdf5":
                    # make sure we only add successful episodes
                    try:
                        with open(os.path.join(root, "ep_stats.json"), "r") as stats_f:
                            ep_stats = json.load(stats_f)

                        if not ep_stats["success"]:
                            continue

                        # if the qa stats already exist, skip
                        qa_stats_path = os.path.join(root, "qa_stats.json")
                        if os.path.exists(qa_stats_path):
                            # check if the manual qa stats also exist
                            with open(qa_stats_path, "r") as f:
                                all_qa_stats = json.load(f)
                            if "manual_qa" in all_qa_stats:
                                continue

                        if args.from_date is not None:
                            ### check that the demo date is from date yyyy-mm-dd onwards ###
                            folder_date = root.split("/")[-3][:10]
                            if folder_date < args.from_date:
                                continue
                    except:
                        continue

                    ep_list.append(os.path.join(root, file))

    for ep_num, ep_path in enumerate(ep_list):
        print(
            colored(
                f"[{ep_num+1}/{len(ep_list)}] Playing episode at: {ep_path}", "yellow"
            )
        )
        manual_inspect_ep(ep_path, video_height=args.video_height)
