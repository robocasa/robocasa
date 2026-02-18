from robocasa.environments.kitchen.kitchen import *


class PrepareSandwichStation(Kitchen):
    """
    Prepare Sandwich Station: composite task for Preparing Sandwiches.

    Simulates the task of gathering sandwich ingredients from the fridge
    and placing them on the counter near the toaster oven for easy access.

    Steps:
        1) Pick up the bowl containing tomato slice, pickle slice, and turkey slice from the fridge.
        2) Pick up the baguette from the fridge.
        3) Place both the bowl and baguette on the counter near the toaster oven.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )

        if "refs" in self._ep_meta:
            self.bowl_rack_level = self._ep_meta["refs"]["bowl_rack_level"]
            self.baguette_rack_level = self._ep_meta["refs"]["baguette_rack_level"]
        else:
            self.bowl_rack_level = -1 if self.rng.random() < 0.5 else -2
            self.baguette_rack_level = -2 if self.bowl_rack_level == -1 else -1

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Carry the bowl containing the sandwich ingredients from the fridge "
            "and place it on the counter near the toaster oven. "
            "Next, pick up the baguette from the fridge and place it near the toaster oven as well."
        )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["bowl_rack_level"] = self.bowl_rack_level
        ep_meta["refs"]["baguette_rack_level"] = self.baguette_rack_level

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ingredient_bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.bowl_rack_level,
                    ),
                    size=(0.5, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato_slice",
                obj_groups="tomato_slice",
                placement=dict(
                    object="ingredient_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pickle_slice",
                obj_groups="pickle_slice",
                placement=dict(
                    object="ingredient_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="turkey_slice",
                obj_groups="turkey_slice",
                placement=dict(
                    object="ingredient_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="baguette",
                obj_groups="baguette",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.baguette_rack_level,
                    ),
                    size=(0.5, 0.27),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        ingredients_in_bowl = []
        for ingredient in ["tomato_slice", "pickle_slice", "turkey_slice"]:
            if OU.check_obj_in_receptacle(self, ingredient, "ingredient_bowl"):
                ingredients_in_bowl.append(ingredient)

        if not ingredients_in_bowl:
            return False

        all_ingredients_ok = True
        for ingredient in ["tomato_slice", "pickle_slice", "turkey_slice"]:
            if ingredient in ingredients_in_bowl:
                continue

            touching_ingredient_in_bowl = False
            for ingredient_in_bowl in ingredients_in_bowl:
                if self.check_contact(
                    self.objects[ingredient], self.objects[ingredient_in_bowl]
                ):
                    touching_ingredient_in_bowl = True
                    break

            if not touching_ingredient_in_bowl:
                if not OU.check_obj_in_receptacle(self, ingredient, "ingredient_bowl"):
                    all_ingredients_ok = False
                    break

        if not all_ingredients_ok:
            return False

        bowl_on_counter = OU.check_obj_any_counter_contact(self, "ingredient_bowl")
        baguette_on_counter = OU.check_obj_any_counter_contact(self, "baguette")

        max_distance = 0.35
        bowl_close_to_toaster = (
            OU.obj_fixture_bbox_min_dist(self, "ingredient_bowl", self.toaster_oven)
            < max_distance
        )
        baguette_close_to_toaster = (
            OU.obj_fixture_bbox_min_dist(self, "baguette", self.toaster_oven)
            < max_distance
        )

        gripper_far = OU.gripper_obj_far(
            self, "ingredient_bowl"
        ) and OU.gripper_obj_far(self, "baguette")

        return (
            all_ingredients_ok
            and bowl_on_counter
            and baguette_on_counter
            and bowl_close_to_toaster
            and baguette_close_to_toaster
            and gripper_far
        )
