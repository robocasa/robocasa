from robocasa.environments.kitchen.kitchen import *


class MeatSkewerAssembly(Kitchen):
    """
    Meat Skewer Assembly: composite task for Filling Serving Dishes activity.

    Simulates the task of preparing kebab skewers for oven cooking by moving them
    from the plate to an oven tray.

    Steps:
        Move 2 kebab skewers from the plate to an oven tray for oven preparation.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Move the kebab skewers from the plate to the oven tray for oven preparation."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="skewer_plate",
                obj_groups="plate",
                object_scale=1.25,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="skewer1",
                obj_groups="kebab_skewer",
                placement=dict(object="skewer_plate", size=(1.0, 1.0)),
            )
        )

        cfgs.append(
            dict(
                name="skewer2",
                obj_groups="kebab_skewer",
                placement=dict(object="skewer_plate", size=(1.0, 1.0)),
            )
        )

        cfgs.append(
            dict(
                name="oven_tray",
                obj_groups="oven_tray",
                object_scale=1.05,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=("ref", "ref"),
                    rotation=(0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        skewer1_on_tray = OU.check_obj_in_receptacle(self, "skewer1", "oven_tray")
        skewer2_on_tray = OU.check_obj_in_receptacle(self, "skewer2", "oven_tray")

        tray_on_counter = OU.check_obj_fixture_contact(
            self, "oven_tray", self.dining_counter
        )

        gripper_far = OU.gripper_obj_far(self, "skewer1") and OU.gripper_obj_far(
            self, "skewer2"
        )

        return skewer1_on_tray and skewer2_on_tray and tray_on_counter and gripper_far
