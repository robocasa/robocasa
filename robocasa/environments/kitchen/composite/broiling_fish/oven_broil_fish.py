from robocasa.environments.kitchen.kitchen import *


class OvenBroilFish(Kitchen):
    """
    Broil Fish: composite task for Broiling Fish activity.

    Simulates preparing the oven for broiling fish.

    Steps:
        1. Place the fish on the tray.
        2. Slide the tray back inside the oven.
        3. Close the oven door.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.oven)
        )

        if "rack_level" in self._ep_meta:
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta["lang"] = (
                f"Place the fish on the {rack_pos} rack of the oven, and slide the tray back inside. "
                "Close the oven door."
            )
        else:
            ep_meta["lang"] = (
                "Place the fish on the rack of the oven, and slide the rack back inside. "
                "Close the oven door."
            )
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_timer = 0
        self.oven.open_door(self)
        self.oven.slide_rack(self, rack_level=self.rack_level)

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
                        ref=self.oven,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        if self.success_timer >= 10:
            return True

        fish_on_rack = self.oven.check_rack_contact(
            self, "obj", rack_level=self.rack_level
        )
        door_closed = self.oven.is_closed(self)

        if door_closed and fish_on_rack:
            self.success_timer += 1
        else:
            self.success_timer = 0

        return False
