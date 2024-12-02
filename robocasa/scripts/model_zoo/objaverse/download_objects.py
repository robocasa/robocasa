import objaverse
import shutil
import bpy
import os


"""
This script downloads objaverse objects as specified by there unique ID (instead of based on a search term)
It needs to be run within Blender
"""


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


if __name__ == "__main__":
    uids = [
        "38f0889da545459093cdc411a0e7ae99",
        "cd1e63b16f804638873d21b822a23617",
        "f77a7095794e460a8ba0b2a55f20a27c",
        "ed137be2b3f74b6bba9181902bd91bf8",
        "e2da46f23af54a7ca5bcaf7581dbfa6b",
    ]
    models_folder = "/Users/lancezhang/projects/kitchen/robosuite-model-zoo-dev/robosuite_model_zoo/scripts/objaverse/utils/download"
    clear_scene()

    # download objects
    objects = objaverse.load_objects(uids)

    for uid, glb_path in objects.items():
        visuals_folder = os.path.join(models_folder, uid[:8], "models")
        os.makedirs(visuals_folder, exist_ok=True)

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
