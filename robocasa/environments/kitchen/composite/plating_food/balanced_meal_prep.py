from robocasa.environments.kitchen.kitchen import *


class BalancedMealPrep(Kitchen):
    """
    Balanced Meal Prep: composite task for Plating Food activity.
    Simulates the task of preparing a balanced meal.

    Steps:
        1. Place the meat on the plate.
        2. Place the vegetable on the same plate.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        vegetable_name = self.get_obj_lang("vegetable")
        ep_meta[
            "lang"
        ] = f"Place the {meat_name} on the plate and place the {vegetable_name} on the same plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.1, 0.1),
                    pos=("ref", -1.0),
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.80, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.80, 0.40),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_on_plate = OU.check_obj_in_receptacle(self, "meat", "plate")
        vegetable_on_plate = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        gripper_far = OU.gripper_obj_far(self, "meat") and OU.gripper_obj_far(
            self, "vegetable"
        )

        return meat_on_plate and vegetable_on_plate and gripper_far
