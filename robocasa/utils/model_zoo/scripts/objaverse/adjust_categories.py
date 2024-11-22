import os
import shutil
import subprocess


source = "../../assets_private/objaverse_final"
copy_to = "../../assets_private/test"
categories = ["banana"]

# set to -1 to adjust all objects in category
items_per_cat = -1


# reset folder
# shutil.rmtree(copy_to)
os.makedirs(copy_to, exist_ok=True)


for cat in categories:
    if "." in cat:
        continue
    cat_folder = os.path.join(source, cat)
    copy_cat_folder = os.path.join(copy_to, cat)
    os.makedirs(copy_cat_folder, exist_ok=True)

    template = "python adjust_object.py --device dummy --mjcf_path {}"
    count = 0
    for obj in os.listdir(cat_folder):
        if "." in obj:
            continue
        obj_folder = os.path.join(cat_folder, obj)
        copy_obj_folder = os.path.join(copy_cat_folder, obj)

        # skip objects that are already adjusted
        if os.path.exists(copy_obj_folder):
            print(copy_obj_folder, "already exists, skipping object")
            continue

        shutil.copytree(obj_folder, copy_obj_folder)

        # now we adjust the object
        command = template.format(os.path.join(copy_obj_folder, "model.xml"))
        subprocess.call(command.split())

        count += 1
        if count == items_per_cat:
            break
