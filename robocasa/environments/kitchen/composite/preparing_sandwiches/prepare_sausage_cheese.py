from robocasa.environments.kitchen.kitchen import *


class PrepareSausageCheese(Kitchen):
    """
    Prepare Sausage and Cheese Sandwich: composite task for Preparing Sandwiches.

    Simulates retrieving sausage and cheese from the fridge and placing them on the cutting board
    for slicing, while bread and a knife are already set next to the board.

    Steps:
        1) Retrieve sausage and cheese from the fridge.
        2) Place them on the cutting board.
        3) Ensure bread and knife are next to the cutting board for slicing later.
    """

    def __init__(
        self, obj_registries=("aigen", "objaverse", "lightwheel"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.9, 0.6), full_depth_region=True),
        )

        self.init_robot_base_ref = self.fridge

        if "refs" in self._ep_meta:
            self.sausage_rack_index = self._ep_meta["refs"]["sausage_rack_index"]
            self.cheese_rack_index = self._ep_meta["refs"]["cheese_rack_index"]
        else:
            self.sausage_rack_index = -1 if self.rng.random() < 0.5 else -2
            self.cheese_rack_index = -2 if self.sausage_rack_index == -1 else -1

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Retrieve the sausage and cheese from the fridge and place them on the cutting board on the counter."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["sausage_rack_index"] = self.sausage_rack_index
        ep_meta["refs"]["cheese_rack_index"] = self.cheese_rack_index
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="sausage",
                obj_groups="sausage",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.sausage_rack_index,
                    ),
                    size=(0.20, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.cheese_rack_index,
                    ),
                    size=(0.15, 0.15),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="baguette",
                obj_groups="baguette",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(0.65, 0.55),
                    pos=(0, -1.0),
                    rotation=(0),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="baguette_container",
                    size=(0.35, 0.50),
                    pos=(0.3, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        ingredients = ["sausage", "cheese"]
        all_ingredients_on_board = all(
            OU.check_obj_in_receptacle(self, item, "baguette_container")
            for item in ingredients
        )

        all_ingredients_far = all(
            OU.gripper_obj_far(self, obj_name=item) for item in ingredients
        )

        return all_ingredients_on_board and all_ingredients_far
