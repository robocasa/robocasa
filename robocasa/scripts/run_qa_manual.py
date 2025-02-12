import argparse
from termcolor import colored
import os
import json
import cv2
import h5py
import time

from robocasa.scripts.run_qa_auto import auto_inspect_ep

COMMON_FAULTS = [
    dict(
        codename="jerky",
        description="robot motion is too jerky",
    ),
    dict(
        codename="too_many_retries",
        description="robot tries to manipulate an object unsuccessfully too many times",
    ),
    dict(
        codename="unnec_base",
        description="unnecessary usage of base",
    ),
    dict(
        codename="slow_or_pause",
        description="robot moving too slowly or having long abnormal pauses",
    ),
    dict(
        codename="misc",
        description="other issue",
    ),
]

FAULT_OPTIONS_MSG = ""
for i, fault in enumerate(COMMON_FAULTS):
    FAULT_OPTIONS_MSG += f"[{i}] {fault['codename']}: {fault['description']}\n"


def play_mp4_opencv(file_path):
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
        resized_frame = cv2.resize(frame, (800 * 3, 800), interpolation=cv2.INTER_AREA)
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

    issue_codes = None
    if user_issues is None:
        issue_codes = None
    else:
        issue_codes = [COMMON_FAULTS[idx]["codename"] for idx in user_issues]

    all_qa_stats["manual_qa"] = dict(
        status=status,
        issues=issue_codes,
        notes=user_notes,
    )

    with open(qa_stats_path, "w") as f:
        json.dump(all_qa_stats, f, indent=4)


def manual_inspect_ep(ds_path):
    print(colored(f"Playing episode at: {ds_path}", "yellow"))
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
    time.sleep(1)
    while True:
        play_mp4_opencv(qa_video)

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
    args = parser.parse_args()

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
                    except:
                        continue

                    ep_list.append(os.path.join(root, file))

    for ep_path in ep_list:
        manual_inspect_ep(ep_path)
