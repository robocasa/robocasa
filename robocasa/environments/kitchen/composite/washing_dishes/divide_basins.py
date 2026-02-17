from robocasa.environments.kitchen.kitchen import *


class DivideBasins(Kitchen):
    """
    Divide Basins: a composite task for Washing Dishes activity.

    One object begins in the right basin, representing an item that is already drying.
    Another object begins on the counter beside the sink and must be moved to the left basin for washing.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_to_wash_lang = self.get_obj_lang("obj_to_wash")
        obj_drying_lang = self.get_obj_lang("obj_drying")

        ep_meta["lang"] = (
            f"Move any existing items from the left basin to the right basin for drying. "
            f"Then move the {obj_to_wash_lang} from the counter to the left basin of the sink for washing."
        )
        ep_meta["obj_to_wash_basin"] = "left"
        ep_meta["obj_drying_basin"] = "right"
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")

    def _get_obj_cfgs(self):
        cfgs = []

        # Preselect 2 small unique items from stackable category
        drink_items = ["cup", "mug", "glass_cup", "measuring_cup"]
        selected_items = self.rng.choice(drink_items, size=2, replace=False)

        cfgs.append(
            dict(
                name="obj_to_wash",
                obj_groups=selected_items[0],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj_drying",
                obj_groups=selected_items[1],
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.4, 0.4),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        in_left_basin = (
            self.sink.get_obj_basin_loc(self, "obj_to_wash", self.sink) == "left"
        )
        in_right_basin = (
            self.sink.get_obj_basin_loc(self, "obj_drying", self.sink) == "right"
        )
        obj_names = ["obj_to_wash", "obj_drying"]
        gripper_far = all(OU.gripper_obj_far(self, obj_name=name) for name in obj_names)
        return in_left_basin and in_right_basin and gripper_far
