import numpy as np
import objaverse
import shutil
import json
import bpy
import os


# list the images in folder and sort them based on their names
def list_objects(category_folder):
    objects = os.listdir(category_folder)
    objects = [obj for obj in objects if "png" in obj]
    object_nums = [int(obj.split(".")[0]) for obj in objects]
    object_nums = np.argsort(object_nums)
    return np.array(objects)[object_nums]


# clears scene in Blender, deletes all objects and images
def clear_scene():
    # delete all objects
    for obj in bpy.data.objects:
        if obj.name != "Light" and obj.name != "Camera":
            obj.select_set(True)
    bpy.ops.object.delete()

    # delete all images
    for image in bpy.data.images:
        if "Image" in image.name:
            bpy.data.images.remove(image)


# checks .mtl file to check (and correct) texture linkages
def check_mtl(file_path):
    # not a perfect method, but functions for most models, potential for improvements
    with open(file_path, "r") as f:
        lines = f.readlines()
    visuals_folder = os.path.dirname(file_path)
    texture_exists = "image0.jpg" in os.listdir(visuals_folder)

    if texture_exists:
        affected = False
        for i in range(len(lines) - 1, -1, -1):
            if "illum" in lines[i]:
                if (i == len(lines) - 1) or ("image" not in lines[i + 1]):
                    lines.insert(i + 1, "map_Kd image0.jpg\n")
                    affected = True

        if affected:
            print("Modified:", file_path)
            with open(file_path, "w") as f:
                contents = "".join(lines)
                f.write(contents)


# converts the format of all objects as specified
def convert_objects(category_folder, objects, force_reconvert=False):
    models_folder = os.path.join(category_folder, "models")
    if os.path.exists(models_folder) and not force_reconvert:
        return

    os.makedirs(models_folder, exist_ok=True)
    clear_scene()

    for obj_meta in objects:
        # create folder for visuals
        visuals_folder = os.path.join(models_folder, obj_meta["uid"][:8], "models")
        os.makedirs(visuals_folder, exist_ok=True)

        # download model
        glb_path = objaverse.load_objects([obj_meta["uid"]]).values()
        glb_path = list(glb_path)[0]

        # import model into blender
        bpy.ops.import_scene.gltf(filepath=glb_path)

        # save all textures
        for i, image in enumerate(bpy.data.images):
            if "Image" in image.name:
                image_save_path = os.path.join(visuals_folder, f"image{i}.jpg")
                image.filepath = image_save_path
                image.save()

        # saving model
        shutil.copyfile(glb_path, os.path.join(visuals_folder, "model.glb"))
        object_save_path = os.path.join(visuals_folder, "model_normalized.obj")
        bpy.ops.export_scene.obj(filepath=object_save_path, path_mode="COPY")

        # check that .mtl file links textures correctly (temporary solution)
        mtl_path = os.path.join(visuals_folder, "model_normalized.mtl")
        check_mtl(mtl_path)

        # delete model and images
        clear_scene()


if __name__ == "__main__":
    root_folder = "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/scripts/objaverse/temp"
    # categories = "syrup chocolate water_bottle bottled_water more_ketchup more_cake bar more_bagel yogurt boxed_drink more_milk bagged_food bottled_drink boxed_food drumstick canned_food beer wine waffle more_steak condiments".split()
    categories = ["more_steak", "waffle", "condiment"]
    force_reconvert = True

    for cat in categories:
        if cat.startswith("."):
            continue
        if cat not in os.listdir(root_folder):
            print('The "' + cat + '"category does not exists, skipping')
            continue

        category_folder = os.path.join(root_folder, cat)
        images = list_objects(category_folder)
        with open(os.path.join(category_folder, "matches.json"), "r") as f:
            matches = json.load(f)

        objects = list()
        for img in images:
            obj_idx = int(img.split(".")[0])
            objects.append(matches[obj_idx])

        convert_objects(category_folder, objects, force_reconvert)
