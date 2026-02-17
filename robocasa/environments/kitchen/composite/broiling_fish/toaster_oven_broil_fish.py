from robocasa.environments.kitchen.kitchen import *


class ToasterOvenBroilFish(Kitchen):
    """
    Broil Fish: composite task for Broiling Fish activity.

    Simulates preparing the toaster oven for broiling fish.

    Steps:
        1. Place the fish on the tray.
        2. Slide the tray back inside the toaster oven.
        3. Close the toaster oven door.
        4. Set the temperature to the max to begin broiling the fish.
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

        if "refs" in self._ep_meta:
            self.rack_level = self._ep_meta["refs"]["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if "rack" in self.chosen_toaster_receptacle:
            receptacle_type = "rack"
        else:
            receptacle_type = "tray"

        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta["lang"] = (
                f"Place the fish on the {rack_pos} {receptacle_type} of the toaster oven, and slide the {receptacle_type} back inside. "
                "Close the toaster oven door and set the temperature to the max to begin broiling the fish."
            )
        else:
            ep_meta["lang"] = (
                f"Place the fish on the {receptacle_type} of the toaster oven, and slide the {receptacle_type} back inside. "
                "Close the toaster oven door and set the temperature to the max to begin broiling the fish."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_timer = 0
        self.toaster_oven.open_door(self)
        self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
            self, rack_level=self.rack_level
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("fish"),
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.5, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fish_on_rack = self.toaster_oven.check_rack_contact(
            self, "obj", rack_level=self.rack_level
        )
        door_closed = self.toaster_oven.is_closed(self)
        temp_at_max = self.toaster_oven._temperature >= 0.95

        return door_closed and fish_on_rack and temp_at_max
