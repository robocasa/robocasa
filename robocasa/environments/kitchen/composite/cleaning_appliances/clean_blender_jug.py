from robocasa.environments.kitchen.kitchen import *


class CleanBlenderJug(Kitchen):
    """
    Clean Blender Jug: composite task for Cleaning Appliances activity.

    Simulates the task of cleaning the blender by taking the blender jug
    and placing it in the sink for washing.

    Steps:
        Take the blender jug from the counter and place it in the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, full_depth_region=True)
        )
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Take the blender jug and place it in the sink to prepare it for cleaning."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="blender_jug",
                obj_groups="blender_jug",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(0.50, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        blender_jug_in_sink = OU.obj_inside_of(self, "blender_jug", self.sink)
        gripper_far_jug = OU.gripper_obj_far(self, "blender_jug")

        return blender_jug_in_sink and gripper_far_jug
