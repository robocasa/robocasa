# RUN SCRIPT IN BLENDER CONSOLE

import bpy
import os


def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == "CONSOLE":
                override = {"window": window, "screen": screen, "area": area}
                bpy.ops.console.scrollback_append(
                    override, text=str(data), type="OUTPUT"
                )


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


clear_scene()

# SET PATH TO MODEL FOLDER HERE - THE FOLDER MUST INCLUDE A .glb / .gltf FILE
model_folder = ""

glb_files = [
    os.path.join(model_folder, name)
    for name in os.listdir(model_folder)
    if name.endswith(".glb") or name.endswith(".gltf")
]

glb_path = glb_files[0]

raw = os.path.join(model_folder, "raw")
os.mkdir(raw)

bpy.ops.import_scene.gltf(filepath=glb_path)

# save all textures
for i, image in enumerate(bpy.data.images):
    if "Image" in image.name:
        image_save_path = os.path.join(raw, f"image{i}.jpg")
        image.filepath = image_save_path
        image.save()
