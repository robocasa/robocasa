from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_doors import OpenOven


class RemoveBroiledFish(OpenOven):
    """
    Remove Broiled Fish: composite task for Broiling Fish activity.

    Simulates the process of removing the broiled fish from the oven and placing it on a plate.

    Steps:
        Take the fish from the oven and place it on the plate on the counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.oven),
        )

        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Take the fish from the oven and place it on the plate on the counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="fish",
                obj_groups="fish",
                object_scale=0.90,
                placement=dict(
                    fixture=self.oven,
                    size=(0.4, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.oven),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fish_on_plate = OU.check_obj_in_receptacle(self, "fish", "plate")
        gripper_fish_far = OU.gripper_obj_far(self, "fish")

        return fish_on_plate and gripper_fish_far
