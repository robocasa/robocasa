from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type, OBJ_CATEGORIES


class MicrowaveCorrectMeal(Kitchen):
    """
    Microwave Correct Meal: composite task for Microwaving Food activity.

    Simulates the task of microwaving the correct meal from two bowls containing
    different food items. One bowl is randomly selected to be microwaved.

    Steps:
        1. Identify which bowl should be microwaved (randomly selected)
        2. Place the selected bowl in the microwave
        3. Close the microwave door
        4. Press the start button to begin microwaving
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave, size=(0.8, 0.5))
        )
        self.init_robot_base_ref = self.counter

        if "refs" in self._ep_meta:
            self.target_bowl = self._ep_meta["refs"]["target_bowl"]
        else:
            self.target_bowl = int(self.rng.choice([0, 1]))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.target_bowl == 0:
            food0_bowl0_lang = self.get_obj_lang("food0_bowl0")
            food1_bowl0_lang = self.get_obj_lang("food1_bowl0")
            target_food_lang = f"{food0_bowl0_lang} and {food1_bowl0_lang}"
        else:
            food0_bowl1_lang = self.get_obj_lang("food0_bowl1")
            food1_bowl1_lang = self.get_obj_lang("food1_bowl1")
            target_food_lang = f"{food0_bowl1_lang} and {food1_bowl1_lang}"

        ep_meta["lang"] = (
            f"Place the bowl containing the {target_food_lang} in the microwave. "
            "Close the microwave door and press the start button to microwave the meal."
        )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["target_bowl"] = self.target_bowl

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.microwave.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        microwavable_categories = get_cats_by_type(
            types=["vegetable", "meat", "cooked_food"],
            obj_registries=self.obj_registries,
        )

        graspable_microwavable_categories = []
        for cat in microwavable_categories:
            for reg in self.obj_registries:
                if (
                    reg in OBJ_CATEGORIES[cat]
                    and OBJ_CATEGORIES[cat][reg].graspable
                    and OBJ_CATEGORIES[cat][reg].microwavable
                ):
                    graspable_microwavable_categories.append(cat)
                    break

        selected_foods = self.rng.choice(
            graspable_microwavable_categories, size=4, replace=False
        )

        cfgs.append(
            dict(
                name="bowl0",
                obj_groups="bowl",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food0_bowl0",
                obj_groups=selected_foods[0],
                placement=dict(
                    object="bowl0",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1_bowl0",
                obj_groups=selected_foods[1],
                placement=dict(
                    object="bowl0",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="bowl0",
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food0_bowl1",
                obj_groups=selected_foods[2],
                placement=dict(
                    object="bowl1",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1_bowl1",
                obj_groups=selected_foods[3],
                placement=dict(
                    object="bowl1",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        target_bowl_name = f"bowl{self.target_bowl}"

        if self.target_bowl == 0:
            target_food0_name = "food0_bowl0"
            target_food1_name = "food1_bowl0"
        else:
            target_food0_name = "food0_bowl1"
            target_food1_name = "food1_bowl1"

        target_bowl_in_microwave = OU.obj_inside_of(
            self, target_bowl_name, self.microwave
        )

        target_food0_in_bowl = OU.check_obj_in_receptacle(
            self, target_food0_name, target_bowl_name
        )
        target_food1_in_bowl = OU.check_obj_in_receptacle(
            self, target_food1_name, target_bowl_name
        )

        door_closed = self.microwave.is_closed(self)

        if not door_closed:
            return False

        microwave_on = self.microwave.get_state()["turned_on"]
        gripper_far = OU.gripper_obj_far(self, obj_name=target_bowl_name)

        return (
            target_bowl_in_microwave
            and target_food0_in_bowl
            and target_food1_in_bowl
            and microwave_on
            and gripper_far
        )
