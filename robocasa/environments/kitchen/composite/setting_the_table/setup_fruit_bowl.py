from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type


class SetupFruitBowl(Kitchen):
    """
    Set Up a Fruit Bowl for the Table: composite task for Setting The Table activity.

    Simulates the task of gathering fruits from the refrigerator and placing them in a bowl on the dining counter.

    Steps:
        1) Retrieve the fruits from the fridge.
        2) Place the fruits into the bowl.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        fruit1_lang = self.get_obj_lang("fruit1")
        fruit2_lang = self.get_obj_lang("fruit2")

        ep_meta[
            "lang"
        ] = f"Retrieve the {fruit1_lang} and {fruit2_lang} from the fridge and place them into the bowl on the dining table."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        fruit_categories = get_cats_by_type(
            types=["fruit"], obj_registries=self.obj_registries
        )

        selected_fruits = self.rng.choice(fruit_categories, size=2, replace=False)

        cfgs = []

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups=selected_fruits[0],
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=-1),
                    size=(0.4, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups=selected_fruits[1],
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=-2),
                    size=(0.4, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.4, 0.35),
                    pos=(0, -0.9),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit_names = ["fruit1", "fruit2"]

        all_fruits_in_bowl = all(
            OU.check_obj_in_receptacle(self, fruit, "bowl") for fruit in fruit_names
        )

        bowl_on_table = OU.check_obj_fixture_contact(self, "bowl", self.dining_counter)
        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")

        return all_fruits_in_bowl and bowl_on_table and gripper_far
