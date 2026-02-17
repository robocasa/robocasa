from robocasa.environments.kitchen.kitchen import *


class LemonSeasoningFish(Kitchen):
    """
    Lemon Seasoning Fish: composite task for Seasoning Food.

    Simulates the task of retrieving a lemon from the fridge and placing it on a cutting board with a fish.

    Steps:
        Open the refrigerator, pick up the lemon, and place it on the cutting board with the fish.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Retrieve a lemon from the fridge and place it on the cutting board with the fish."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="lemon",
                obj_groups="lemon",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.15, 0.10),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fish",
                obj_groups="fish",
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.5),
                    pos=(0, -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return OU.check_obj_in_receptacle(
            self, "lemon", "fish_container"
        ) and OU.gripper_obj_far(self, "lemon")
