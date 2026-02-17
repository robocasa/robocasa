from robocasa.environments.kitchen.kitchen import *


class PrepareSmoothie(Kitchen):
    """
    PrepareSmoothie: composite task for the making smoothies activity.
    Simulates the task of preparing ingredients for making a smoothie.
    Steps:
        1. Pick the food item for making a fruit smoothie from the fridge.
        2. Place it on the plate next to the blender.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_item = self.get_obj_lang("smoothie_food1")
        food_item_2 = self.get_obj_lang("smoothie_food2")
        if food_item != food_item_2:
            food_lang = f"{food_item} and {food_item_2}"
        else:
            food_lang = f"two {food_item}s"
        ep_meta[
            "lang"
        ] = f"Pick the {food_lang} from the fridge and place them on the plate next to the blender to make a fruit smoothie."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="smoothie_food1",
                obj_groups="fruit",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.3),
                    pos=(0.0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )
        cfgs.append(
            dict(
                name="smoothie_food2",
                obj_groups="fruit",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.3),
                    pos=(0.0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_item",
                exclude_obj_groups=["fruit", "vegetable"],
                placement=dict(
                    fixture=FixtureType.FRIDGE,
                    size=(0.2, 0.2),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.35),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        distr_item_in_fridge = OU.obj_inside_of(self, "distr_item", self.fridge)

        foods_on_plate = OU.check_obj_in_receptacle(
            self, "smoothie_food1", "plate"
        ) and OU.check_obj_in_receptacle(self, "smoothie_food2", "plate")
        gripper_far = OU.gripper_obj_far(self, "smoothie_food1") and OU.gripper_obj_far(
            self, "smoothie_food2"
        )
        return foods_on_plate and gripper_far and distr_item_in_fridge
