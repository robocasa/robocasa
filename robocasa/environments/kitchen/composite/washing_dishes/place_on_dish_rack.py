from robocasa.environments.kitchen.kitchen import *


class PlaceOnDishRack(Kitchen):
    """
    Place Clean Dishes on Dish Rack: a composite task for Washing Dishes activity.

    Simulates the task of transferring clean dishes from the right sink basin to the dish rack
    for drying, while leaving dirty items in the left basin untouched.

    Steps:
        1. Identify clean dishes in the right sink basin
        2. Pick up each clean dish from the right basin
        3. Place them on the dish rack for drying
        4. Leave dirty items in the left basin undisturbed
    """

    EXCLUDE_STYLES = [
        2,
        5,
        7,
        8,
        10,
        11,
        13,
        16,
        17,
        18,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        40,
        44,
        45,
        46,
        47,
        48,
        50,
        51,
        53,
        54,
        56,
        57,
        58,
        59,
        60,
    ]

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["dish_rack"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.dish_rack = self.register_fixture_ref(
            "dish_rack", dict(id=FixtureType.DISH_RACK)
        )

        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            f"Transfer the clean items from the right sink basin "
            f"to the dish rack for drying. Leave any dirty items in the left basin untouched."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="clean_item1",
                obj_groups=("mug", "cup"),
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        side="right",
                    ),
                    size=(1.0, 1.0),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="clean_item2",
                obj_groups=("mug", "cup"),
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        side="right",
                    ),
                    size=(1.0, 1.0),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dirty_item1",
                obj_groups=("mug", "cup", "glass_cup"),
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        side="left",
                    ),
                    size=(1.0, 1.0),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dirty_item2",
                obj_groups=("fork", "cup", "spoon"),
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        side="left",
                    ),
                    size=(1.0, 1.0),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        clean_dishes = ["clean_item1", "clean_item2"]
        dirty_dishes = ["dirty_item1", "dirty_item2"]

        clean_on_rack = all(
            OU.check_obj_fixture_contact(self, dish, self.dish_rack)
            for dish in clean_dishes
        )

        clean_not_on_counter = all(
            not OU.check_obj_any_counter_contact(self, dish) for dish in clean_dishes
        )

        dirty_left_in_sink = all(
            self.sink.get_obj_basin_loc(self, dish, self.sink) == "left"
            for dish in dirty_dishes
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=name)
            for name in clean_dishes + dirty_dishes
        )

        return (
            clean_on_rack
            and clean_not_on_counter
            and dirty_left_in_sink
            and gripper_far
        )
