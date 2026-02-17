from robocasa.environments.kitchen.kitchen import *


class RecycleStackedYogurt(Kitchen):
    """
    Recycle Stacked Yogurt: composite task for Organizing Recycling.

    Simulates the task of stacking yogurt containers to prepare for recycling.

    Steps:
        Pick up the second largest yogurt container from the counter and stack it on the largest container.
        Pick up the smaller container and place it on top of the second largest container.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.stool

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Stack the yogurt containers by placing the smaller containers on top of the larger containers."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="yogurt_container1",
                obj_groups="yogurt",
                object_scale=2.0,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.70, 0.3),
                    pos=(0, "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt_container2",
                obj_groups="yogurt",
                object_scale=1.30,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.70, 0.3),
                    pos=(0, "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt_container3",
                obj_groups="yogurt",
                object_scale=0.90,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.70, 0.3),
                    pos=(0, "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        container2_in_container1 = OU.check_obj_in_receptacle(
            self, "yogurt_container2", "yogurt_container1"
        )
        container3_in_container2 = OU.check_obj_in_receptacle(
            self, "yogurt_container3", "yogurt_container2"
        )

        pos1 = self.sim.data.body_xpos[self.obj_body_id["yogurt_container1"]]
        pos2 = self.sim.data.body_xpos[self.obj_body_id["yogurt_container2"]]
        pos3 = self.sim.data.body_xpos[self.obj_body_id["yogurt_container3"]]

        z1, z2, z3 = pos1[2], pos2[2], pos3[2]

        z_thresh = 0.04
        stacked_correctly = (z2 > z1 + z_thresh) and (z3 > z2 + z_thresh)

        gripper_container1_far = OU.gripper_obj_far(self, obj_name="yogurt_container1")
        gripper_container2_far = OU.gripper_obj_far(self, obj_name="yogurt_container2")
        gripper_container3_far = OU.gripper_obj_far(self, obj_name="yogurt_container3")

        return (
            container2_in_container1
            and container3_in_container2
            and stacked_correctly
            and gripper_container1_far
            and gripper_container2_far
            and gripper_container3_far
        )
