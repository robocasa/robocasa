from robocasa.environments.kitchen.kitchen import *


class ReturnHeatedFood(Kitchen):
    """
    Return Heated Food: composite task for Microwaving Food activity.

    Simulates the task of placing a bowl containing heated meat and vegetable
    on the dining counter for serving.

    Steps:
        1. Open the microwave door
        2. Take the bowl containing meat and vegetable from the microwave
        3. Place it on the dining counter
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        vegetable_lang = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"The microwave is finished heating the {meat_lang} and {vegetable_lang}. Take the bowl "
            "containing the food and place it on the dining counter."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        bowl_on_dining_counter = OU.check_obj_fixture_contact(
            self, "bowl", self.dining_counter
        )
        meat_in_bowl = OU.check_obj_in_receptacle(self, "meat", "bowl")
        vegetable_in_bowl = OU.check_obj_in_receptacle(self, "vegetable", "bowl")

        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")

        return (
            bowl_on_dining_counter
            and meat_in_bowl
            and vegetable_in_bowl
            and gripper_far
        )
