import json
import os
import shutil


"""
Checks for object overlaps in the specified folder (both inter- and intra-categories).
Ask for user input to resolve such conflicts.
"""

target_folder = (
    "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/assets_private"
    "/objaverse_combined_refactored"
)


seen = dict()
conflicts = dict()

for cat in os.listdir(target_folder):
    if "." in cat:
        continue
    cat_folder = os.path.join(target_folder, cat)

    for obj in os.listdir(cat_folder):
        if "." in obj:
            continue
        obj_folder = os.path.join(cat_folder, obj)

        with open(os.path.join(obj_folder, "meta.json"), "r") as f:
            meta = json.load(f)

        # this uid is truncated (8 alphanumerics), so might not always work
        uid = os.path.split(meta["path"])[-1]
        if uid not in seen:
            seen[uid] = obj_folder
        else:
            if uid in conflicts:
                conflicts[uid].append(obj_folder)
                obj_folders = conflicts[uid]
            else:
                obj_folders = [obj_folder, seen[uid]]
                conflicts[uid] = obj_folders
            print("Conflict between:")
            for i, obj_folder in enumerate(obj_folders):
                print("\t{}.".format(i), obj_folder)

            while True:
                keep_idx = input("Which one to keep? ")
                try:
                    keep_idx = int(keep_idx)
                    if keep_idx < len(obj_folders):
                        break
                except ValueError:
                    pass
            for i, obj_folder in enumerate(obj_folders):
                if i == keep_idx:
                    continue
                print("Removing:", obj_folder)
                shutil.rmtree(obj_folder)
            print()

print("Total unique objects:", len(seen))
