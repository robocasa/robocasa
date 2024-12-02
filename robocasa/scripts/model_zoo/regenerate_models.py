import argparse
import os
import json
import subprocess
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to model(s)",
        required=True,
    )
    args = parser.parse_args()

    # find all commands to run
    json_paths = []
    for root, dirs, files in os.walk(args.folder):
        if "meta.json" in files:
            json_paths.append(os.path.join(root, "meta.json"))

    commands = []
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        commands.append(data["cmd"])

    processes = set()
    max_processes = 5

    for cmd in commands:
        processes.add(subprocess.Popen(cmd, shell=True))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])
