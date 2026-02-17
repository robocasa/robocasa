from robocasa.environments.kitchen.kitchen import *


class ToastOnCorrectRack(Kitchen):
    """
    ToastOnCorrectRack: composite task for Toasting Bread activity.

    Steps:
        1. Open the toaster oven door.
        2. Pick up the bread and place it on the specified rack inside the toaster oven.
        3. Pick up the meat and place it on the specified rack inside the toaster oven.
    """

    # Exclude styles with 1 rack only
    EXCLUDE_STYLES = [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        22,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        38,
        40,
        41,
        42,
        43,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        56,
        58,
        59,
        60,
    ]

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
        if "bread_rack_level" in self._ep_meta:
            self.bread_rack_level = self._ep_meta["bread_rack_level"]
        else:
            self.bread_rack_level = 1 if self.rng.random() > 0.5 else 0

        if self.bread_rack_level == 0:
            self.meat_rack_level = 1
        else:
            self.meat_rack_level = 0

        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        receptacle_type = (
            "rack" if "rack" in self.toaster_oven.rack_or_tray(self) else "tray"
        )
        bread_lang = self.get_obj_lang("bread")
        meat_lang = self.get_obj_lang("meat")

        if self.bread_rack_level == 0:
            bread_direction = "bottom"
            meat_direction = "top"
        else:
            bread_direction = "top"
            meat_direction = "bottom"

        ep_meta["lang"] = (
            f"Open the toaster oven door. Place the {bread_lang} on the {bread_direction} {receptacle_type} in the toaster oven "
            f"and place the {meat_lang} on the {meat_direction} {receptacle_type}."
        )
        ep_meta["bread_rack_level"] = self.bread_rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.75, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                exclude_obj_groups=("fish", "shrimp"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.75, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        bread_on_rack = self.toaster_oven.check_rack_contact(
            self, "bread", rack_level=self.bread_rack_level
        )
        meat_on_rack = self.toaster_oven.check_rack_contact(
            self, "meat", rack_level=self.meat_rack_level
        )
        obj_gripper_far = OU.gripper_obj_far(self, "bread") and OU.gripper_obj_far(
            self, "meat"
        )
        return bread_on_rack and meat_on_rack and obj_gripper_far
