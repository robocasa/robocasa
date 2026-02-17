from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import *


class CutBuffetPizza(ManipulateDrawer):
    """
    Cut Buffet Pizza: composite task for Arranging Buffet.

    Simulates taking a pizza cutter from a drawer and placing it on the dining counter
    where pizza is already present for cutting.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.DRAWER, ref=self.dining_counter)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            f"Take the pizza cutter from the drawer on the {self.drawer_side} and place it on the "
            f"dining counter for cutting."
        )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pizza_cutter",
                obj_groups="pizza_cutter",
                object_scale=[1, 1, 1.5],
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.5),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pizza",
                obj_groups="pizza",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(0.50, 0.50),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_utensil",
                obj_groups=("spoon", "fork"),
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_utensil_2",
                obj_groups=("spoon", "fork"),
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="distractor_utensil",
                    size=(1.0, 1.0),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_bread_food",
                obj_groups="bread_food",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                    try_to_place_in="tray",
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_mug",
                obj_groups="mug",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_mug_2",
                obj_groups="mug",
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="distractor_mug",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        pizza_cutter_on_counter = OU.check_obj_fixture_contact(
            self, "pizza_cutter", self.dining_counter
        )

        pizza_on_counter = OU.check_obj_fixture_contact(
            self, "pizza", self.dining_counter
        )

        gripper_far = OU.gripper_obj_far(
            self, obj_name="pizza_cutter", th=0.15
        ) and OU.gripper_obj_far(self, obj_name="pizza", th=0.15)

        return pizza_cutter_on_counter and pizza_on_counter and gripper_far
