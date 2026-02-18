from robocasa.environments.kitchen.kitchen import *


class PrepFridgeForCleaning(Kitchen):
    """
    Prep Fridge for Cleaning: composite task for Cleaning Appliances activity.

    Simulates the task of clearing the fridge of perishable items to prepare for interior cleaning.

    Steps:
        Open the fridge, remove all food items (fruit, vegetable, meat), and place them on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit_name = self.get_obj_lang("fruit")
        vegetable_name = self.get_obj_lang("vegetable")
        meat_name = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            f"Open the fridge, remove the {fruit_name}, "
            f"{vegetable_name}, and {meat_name}, "
            f"and place them on any counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):

        cfgs = []
        cfgs.append(
            dict(
                name="fruit",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        fruit_on_counter = OU.check_obj_any_counter_contact(self, "fruit")
        vegetable_on_counter = OU.check_obj_any_counter_contact(self, "vegetable")
        meat_on_counter = OU.check_obj_any_counter_contact(self, "meat")

        gripper_far_fruit = OU.gripper_obj_far(self, "fruit")
        gripper_far_vegetable = OU.gripper_obj_far(self, "vegetable")
        gripper_far_meat = OU.gripper_obj_far(self, "meat")

        return (
            fruit_on_counter
            and vegetable_on_counter
            and meat_on_counter
            and gripper_far_fruit
            and gripper_far_vegetable
            and gripper_far_meat
        )
