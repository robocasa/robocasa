from robocasa.environments.kitchen.kitchen import *


class ToastBaguette(Kitchen):
    """
    Warm-up of Pastries: composite task for Toasting Bread activity.

    Steps:
        1. Slide the toaster oven door out.
        2. Pick up the baguette from the plate on the counter.
        3. Place it on the toaster oven rack.
        4. Close the toaster oven door fully.
        5. Set the temperature to warm (lowest setting).
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
        if "random_temp" in self._ep_meta:
            self.random_temp = self._ep_meta["random_temp"]
        else:
            self.random_temp = float(self.rng.uniform(0.2, 1.0))
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["random_temp"] = self.random_temp
        ep_meta["lang"] = (
            "Pull the toaster oven rack outwards. Pick up the baguette from the plate and place it on the toaster oven rack. "
            "Close the toaster oven door fully and set the temperature to the lowest setting."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_oven.open_door(self)
        self.toaster_oven.set_temperature(self, self.random_temp)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="baguette",
                obj_groups="baguette",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.5, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        baguette_in_toaster = self.toaster_oven.check_rack_contact(
            self, "baguette", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "baguette", rack_level=1)

        if baguette_in_toaster:
            toaster_oven_state = self.toaster_oven.get_state()
            door_closed = self.toaster_oven.is_closed(self)
            return door_closed and toaster_oven_state["temperature"] <= 0.05
        else:
            return False
