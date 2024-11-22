import shutil
import os


"""
Create new dataset based on manual selection file.
File must have one line category name and one line selected/removed object IDs (based on idx in folder name).
"""

src_folder = (
    "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/assets_private"
    "/objaverse_combined_refactored"
)
dst_folder = (
    "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/assets_private"
    "/objaverse_combined_final"
)
selection_file = "selection_texts/final_removed.txt"

# True if specified objects need to be removed
specify_removed = True


# create/clear destination folder
if os.path.exists(dst_folder):
    shutil.rmtree(dst_folder)
os.makedirs(dst_folder)

# read categories and corresponding indices
with open(selection_file, "r") as f:
    lines = f.readlines()
cats = [lines[i * 2].strip() for i in range(len(lines) // 2)]
specified = [lines[i * 2 + 1].strip().split() for i in range(len(lines) // 2)]

# add unspecified categories if necessary
if specify_removed:
    for cat in os.listdir(src_folder):
        if "." in cat or cat in cats:
            continue
        cats.append(cat)

# copy over specified objects
total_objects = 0
for i, cat in enumerate(cats):
    cat_specified = specified[i] if i < len(specified) else list()
    obj_count = 0

    cat_src_folder = os.path.join(src_folder, cat)
    cat_dst_folder = os.path.join(dst_folder, cat)
    os.makedirs(cat_dst_folder)

    for obj in os.listdir(cat_src_folder):
        if "." in obj:
            continue
        obj_idx = obj.split("_")[-1]

        copy = (
            (obj_idx not in cat_specified)
            if specify_removed
            else (obj_idx in cat_specified)
        )
        if copy:
            shutil.copytree(
                os.path.join(cat_src_folder, obj), os.path.join(cat_dst_folder, obj)
            )
            obj_count += 1

    print(cat.ljust(20), obj_count)
    total_objects += obj_count

print("Total objects:", total_objects)
