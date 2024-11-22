import os
import shutil
import json


"""
This script performs two functionalities:
    1. Combines multiple categories into one (need to specify manually) and resolve conflicts
    2. Checks and resolves conflicts within each individual category, re-indexes objects
"""

delete_original = False
source_folder = (
    "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/assets_private"
    "/objaverse_combined_final"
)
target_folder = (
    "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/assets_private"
    "/objaverse_final"
)
rename_original = source_folder == target_folder


# combinations = {
#     "tangerine": ["mandarin", "tangerine"],
#     "knife": ["knife", "kitchen_knife"],
#     "cake": ["cake", "more_cake"],
#     "steak": ["steak", "more_steak"],
#     "canned_food": ["canned_food", "more_canned_food"],
#     "bagel": ["bagel", "more_bagel"],
#     "bagged_food": ["bagged_food", "more_bagged_food"],
#     "fish": ["fish", "more_fish"],
#     "ketchup": ["ketchup", "more_ketchup"],
#     "milk": ["milk", "more_milk"]
# }

# add single categories that don't need to be combined
combinations = dict()
existing = set()
for k, v in combinations.items():
    existing.add(k)
    for cat in v:
        existing.add(cat)

for cat in os.listdir(source_folder):
    if cat in existing:
        continue
    if not os.path.isdir(os.path.join(source_folder, cat)):
        continue
    existing.add(cat)
    combinations[cat] = [cat]


def get_uid(folder):
    with open(os.path.join(folder, "meta.json"), "r") as f:
        meta = json.load(f)
    uid = os.path.split(meta["path"])[-1]
    return uid


print("Source folder:", source_folder)
print("Target folder:", target_folder, "\n")
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)

# Refactors categories
for new_cat_name in combinations.keys():
    idx = 0

    # rename previous folders (if necessary)
    prev_cats = combinations[new_cat_name]
    prev_folders = list()
    for cat in prev_cats:
        source = os.path.join(source_folder, cat)
        if rename_original and not cat.startswith("_"):
            dest = os.path.join(source_folder, "_" + cat)
            shutil.move(source, dest)
            prev_folders.append(dest)
        else:
            prev_folders.append(source)

    # create new folder, clear if already exists
    new_folder = os.path.join(target_folder, new_cat_name)
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
    os.makedirs(new_folder)

    # combine categories
    seen = set()
    idx = 0

    for folder in prev_folders:
        print(os.path.basename(folder), "->", new_cat_name)
        for obj in os.listdir(folder):
            obj_folder = os.path.join(folder, obj)
            if not os.path.isdir(obj_folder):
                continue

            # check for conflict
            uid = get_uid(obj_folder)
            if uid in seen:
                print("Already seen:", obj_folder)
                continue
            else:
                seen.add(uid)

            # rename object and copy over
            new_obj_name = new_cat_name + "_" + str(idx)
            shutil.copytree(obj_folder, os.path.join(new_folder, new_obj_name))

            idx += 1
        # delete original categories if needed
        if delete_original:
            shutil.rmtree(folder)
        print()
