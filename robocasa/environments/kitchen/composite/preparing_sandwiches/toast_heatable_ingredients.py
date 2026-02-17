from robocasa.environments.kitchen.kitchen import *


class ToastHeatableIngredients(Kitchen):
    """
    Toast Heatable Ingredients: composite task for Preparing Sandwiches.

    Simulates the task of toasting heatable sandwich ingredients in the toaster oven.

    Steps:
        1) Move the heatable items to the toaster oven rack.
        2) Make sure the cold or unheable items are still on the plate.
    """

    def __init__(
        self,
        obj_registries=("aigen", "lightwheel", "objaverse"),
        enable_fixtures=None,
        *args,
        **kwargs
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(
            obj_registries=obj_registries,
            enable_fixtures=enable_fixtures,
            *args,
            **kwargs
        )

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
        ep_meta["lang"] = (
            "Open the toaster oven door and place the heatable sandwich ingredients on the rack only. "
            "Make sure the cold/unheable items are still on the plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.5,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.75, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sausage",
                obj_groups="sausage",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread",
                obj_groups="baguette",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sausage_in_toaster = self.toaster_oven.check_rack_contact(
            self, "sausage", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "sausage", rack_level=1)
        bread_in_toaster = self.toaster_oven.check_rack_contact(
            self, "bread", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "bread", rack_level=1)

        tomato_still_on_plate = OU.check_obj_in_receptacle(self, "tomato", "plate")
        onion_still_on_plate = OU.check_obj_in_receptacle(self, "onion", "plate")

        gripper_far = (
            OU.gripper_obj_far(self, "sausage", th=0.15)
            and OU.gripper_obj_far(self, "bread", th=0.15)
            and OU.gripper_obj_far(self, "tomato", th=0.15)
            and OU.gripper_obj_far(self, "onion", th=0.15)
            and OU.gripper_obj_far(self, "plate", th=0.15)
        )

        return (
            sausage_in_toaster
            and bread_in_toaster
            and tomato_still_on_plate
            and onion_still_on_plate
            and gripper_far
        )
