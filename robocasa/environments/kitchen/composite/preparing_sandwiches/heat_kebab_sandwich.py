from robocasa.environments.kitchen.kitchen import *


class HeatKebabSandwich(Kitchen):
    """
    Heat Kebab Sandwich: composite task for Preparing Sandwiches.

    Simulates the task of heating up a kebab sandwich in the toaster oven.

    Steps:
        1) Pick up the kebab skewer and bread, and place them inside the toaster oven.
        2) Once inside, close the toaster oven door.
        3) Set the timer for at least 10% of the total time.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the kebab skewer and baguette bread, and place them inside the toaster oven. "
                "Close the toaster oven door and start by setting the timer."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.toaster_oven.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.25,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="kebab",
                obj_groups="kebab_skewer",
                object_scale=[1.1, 0.8, 1],
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="baguette",
                obj_groups="baguette",
                placement=dict(
                    object="plate",
                    size=(0.9, 0.9),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        baguette_in_toaster = self.toaster_oven.check_rack_contact(
            self, "baguette", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "baguette", rack_level=1)

        kebab_in_toaster = self.toaster_oven.check_rack_contact(
            self, "kebab", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "kebab", rack_level=1)
        toaster_oven_state = self.toaster_oven.get_state()

        if baguette_in_toaster and kebab_in_toaster:
            toaster_oven_state = self.toaster_oven.get_state(self)
            if self.toaster_oven.is_open(self):
                return False
            return toaster_oven_state["time"] >= 0.1
        else:
            return False
