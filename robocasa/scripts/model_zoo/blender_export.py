"""
Initial code from https://blender.stackexchange.com/a/150922
"""

import bpy

from bpy.props import (
    BoolProperty,
    FloatProperty,
    StringProperty,
    CollectionProperty,
    EnumProperty,
)
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os
import subprocess

# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
class EXPORT_OT_customOBJ(Operator, ExportHelper):
    """Export the scene to Obj"""

    bl_idname = "export_scene.custom_obj"
    bl_label = "Export Obj"

    # ExportHelper mixin class uses this
    filename_ext = ""  # .obj

    filter_glob: StringProperty(
        default="*.obj",
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.

    axis_forward: EnumProperty(
        name="Axis Forward",
        description="Axis Forward",
        items={
            ("X", "X Forward", "X"),
            ("Y", "Y Forward", "Y"),
            ("Z", "Z Forward", "Z"),
            ("-X", "-X Forward", "-X"),
            ("-Y", "-Y Forward", "-Y"),
            ("-Z", "-Z Forward", "-Z"),
        },
        default="X",
    )

    axis_up: EnumProperty(
        name="Axis Up",
        description="Axis Up",
        items={
            ("X", "X Up", "X"),
            ("Y", "Y Up", "Y"),
            ("Z", "Z Up", "Z"),
            ("-X", "-X Up", "-X"),
            ("-Y", "-Y Up", "-Y"),
            ("-Z", "-Z Up", "-Z"),
        },
        default="Z",
    )

    def execute(self, context):

        # mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
        #
        # for obj in mesh:
        #     obj.select_set(state=True)
        #
        #     bpy.context.view_layer.objects.active = obj
        #
        # bpy.ops.object.join()

        basedir = self.filepath
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # Deselect all objects
        bpy.ops.obj.select_all(action="DESELECT")

        # loop through all the objects in the scene
        scene = bpy.context.scene
        for ob in scene.objects:
            # Select each object
            ob.select_set(True)

            # Make sure that we only export meshes
            if ob.type == "MESH":
                # Export the currently selected object to its own file based on its name
                bpy.ops.export_scene.obj(
                    filepath=os.path.join(basedir, ob.name + ".obj"),
                    use_selection=True,
                    global_scale=1.0,  # self.scale,
                    axis_forward=self.axis_forward,
                    axis_up=self.axis_up,
                )
            # Deselect the object and move on to another if any more are left
            ob.select_set(False)

        # # Export
        # bpy.ops.export_scene.obj(
        #     filepath=self.filepath,
        #     global_scale=1.0, #self.scale,
        #     axis_forward=self.axis_forward,
        #     axis_up=self.axis_up,
        # )

        # Undo!
        # num_undos = 2
        # for i in range(num_undos):
        #     print("undoing", i)
        #     bpy.ops.ed.undo()
        # bpy.ops.ed.undo()

        return {"FINISHED"}


class ExportObjAndMJCF(Operator, ExportHelper):
    """Export the scene to OBJ and MJCF"""

    bl_idname = "export_scene.obj_and_mjcf"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Obj and MJCF"

    # ExportHelper mixin class uses this
    filename_ext = ".obj"

    filter_glob: StringProperty(
        default="*.obj",
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.

    axis_forward: EnumProperty(
        name="Axis Forward",
        description="Axis Forward",
        items={
            ("X", "X Forward", "X"),
            ("Y", "Y Forward", "Y"),
            ("Z", "Z Forward", "Z"),
            ("-X", "-X Forward", "-X"),
            ("-Y", "-Y Forward", "-Y"),
            ("-Z", "-Z Forward", "-Z"),
        },
        default="X",
    )

    axis_up: EnumProperty(
        name="Axis Up",
        description="Axis Up",
        items={
            ("X", "X Up", "X"),
            ("Y", "Y Up", "Y"),
            ("Z", "Z Up", "Z"),
            ("-X", "-X Up", "-X"),
            ("-Y", "-Y Up", "-Y"),
            ("-Z", "-Z Up", "-Z"),
        },
        default="Z",
    )

    scale: FloatProperty(
        name="Scale",
        description="Scale",
        default=1.0,
    )

    show_coll_geoms: BoolProperty(
        name="Show Collision Geoms",
        description="Display collision geoms in robosuite",
        default=False,
    )

    hide_vis_geoms: BoolProperty(
        name="Hide Visual Geoms",
        description="No visual geoms in robosuite",
        default=False,
    )

    model_name: StringProperty(
        name="MJCF Model Name",
        description="(optional) name of saved mjcf model. Defaults to name from obj filename",
        default="",
        maxlen=255,
    )

    def execute(self, context):

        # mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
        #
        # for obj in mesh:
        #     obj.select_set(state=True)
        #
        #     bpy.context.view_layer.objects.active = obj
        #
        # bpy.ops.object.join()

        # Export
        bpy.ops.export_scene.obj(
            filepath=self.filepath,
            global_scale=1.0,  # self.scale,
            axis_forward=self.axis_forward,
            axis_up=self.axis_up,
        )

        conda_command = "source activate {}".format(os.environ["CONDA_PREFIX"])
        repo_path_command = 'python -c \\"import robosuite_model_zoo; print(robosuite_model_zoo.__path__[0])\\"'
        repo_path = (
            subprocess.check_output(
                'bash -c "{}; {}"'.format(conda_command, repo_path_command), shell=True
            )
            .decode("UTF-8")
            .rstrip("\n")
        )

        command_to_run = "python {script_path} --model_path {model_path} --scale {scale} --verbose".format(
            script_path=os.path.join(repo_path, "scripts/generate_object_model.py"),
            # script_path="/Users/soroushnasiriany/research/robosuite-model-zoo-dev/robosuite_model_zoo/scripts/generate_object_model.py",
            model_path=self.filepath,
            scale=self.scale,
        )
        if self.show_coll_geoms:
            command_to_run += " --show_coll_geoms"
        if self.hide_vis_geoms:
            command_to_run += " --hide_vis_geoms"
        if len(self.model_name) > 0:
            command_to_run += " --model_name {}".format(self.model_name)

        subprocess.run(
            'bash -c "{}; {}"'.format(conda_command, command_to_run), shell=True
        )

        # Undo!
        # num_undos = 2
        # for i in range(num_undos):
        #     print("undoing", i)
        #     bpy.ops.ed.undo()
        # bpy.ops.ed.undo()

        return {"FINISHED"}


# Only needed if you want to add into a dynamic menu
def draw_export_obj(self, context):
    self.layout.operator(
        EXPORT_OT_customOBJ.bl_idname, text="Robosuite Obj", icon="MESH_MONKEY"
    )


def draw_export_obj_and_mjcf(self, context):
    self.layout.operator(
        ExportObjAndMJCF.bl_idname, text="Robosuite Obj and MJCF", icon="MESH_MONKEY"
    )


# Registration
classes = (
    EXPORT_OT_customOBJ,
    ExportObjAndMJCF,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    bpy.types.TOPBAR_MT_file_export.prepend(draw_export_obj_and_mjcf)
    bpy.types.TOPBAR_MT_file_export.prepend(draw_export_obj)


def unregister():
    from bpy.utils import unregister_class

    for cls in reversed(classes):
        unregister_class(cls)

    bpy.types.TOPBAR_MT_file_export.remove(draw_export_obj_and_mjcf)
    bpy.types.TOPBAR_MT_file_export.remove(draw_export_obj)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.export_scene.custom_obj('INVOKE_DEFAULT')
    # bpy.ops.export_scene.obj_and_mjcf('INVOKE_DEFAULT')
