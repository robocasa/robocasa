from robocasa.environments.kitchen.kitchen import *


class AddMarshmallow(Kitchen):
    """
    Add Marshmallow: composite task for Preparing Hot Chocolate activity.

    Simulates the task of adding marshmallow to a mug with hot chocolate.

    Steps:
        Pick the marshmallow, go to the dining counter with hot chocolate,
        and place it in the hot chocolate.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, full_depth_region=True)
        )

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f" Pick up the marshmallow and place it in the cup of hot chocolate on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "mug", [0.36, 0.20, 0.09, 1.0])

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"marshmallow",
                obj_groups="marshmallow",
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.4),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        top_size=(0.4, 0.4), full_depth_region=True
                    ),
                    try_to_place_in="plate",
                ),
                init_robot_here=True,
            )
        )

        cfgs.append(
            dict(
                name="mug",
                obj_groups="mug",
                placement=dict(
                    fixture=self.dining_counter,
                    pos=("ref", "ref"),
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.2, 0.2),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        marshmallow_in_mug = OU.check_obj_in_receptacle(self, "marshmallow", "mug")
        gripper_far = OU.gripper_obj_far(self, "mug")
        return marshmallow_in_mug and gripper_far
