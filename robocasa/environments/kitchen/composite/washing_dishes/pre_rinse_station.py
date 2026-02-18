from robocasa.environments.kitchen.kitchen import *


class PreRinseStation(Kitchen):
    """
    Pre-Rinse Station Task: composite task for Washing Dishes activity.

    Simulates the task of sorting dishes into two separate basins of a double sink.
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
        41,  # right basin is too small
        43,  # right basin is too small
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

        # Determine object-to-basin assignments randomly
        if "refs" in self._ep_meta:
            self.left_obj = self._ep_meta["refs"]["left_obj"]
            self.right_obj = self._ep_meta["refs"]["right_obj"]
        else:
            if self.rng.random() < 0.5:
                self.left_obj, self.right_obj = "receptacle", "stackable"
            else:
                self.left_obj, self.right_obj = "stackable", "receptacle"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        left_lang = self.get_obj_lang(obj_name=self.left_obj)
        right_lang = self.get_obj_lang(obj_name=self.right_obj)
        ep_meta[
            "lang"
        ] = f"Place the {left_lang} in the left basin and the {right_lang} in the right basin of the sink."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["left_obj"] = self.left_obj
        ep_meta["refs"]["right_obj"] = self.right_obj

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.water_running = False
        self.water_running_steps = 0
        self.steps_elapsed = 0

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.left_obj,
                obj_groups=self.left_obj,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            ),
            dict(
                name=self.right_obj,
                obj_groups=self.right_obj,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            ),
        ]

    def _check_success(self):
        left_ok = self.sink.get_obj_basin_loc(self, self.left_obj, self.sink) == "left"
        right_ok = (
            self.sink.get_obj_basin_loc(self, self.right_obj, self.sink) == "right"
        )

        gripper_far_left = OU.gripper_obj_far(self, obj_name=self.left_obj)
        gripper_far_right = OU.gripper_obj_far(self, obj_name=self.right_obj)

        return left_ok and right_ok and gripper_far_left and gripper_far_right
