from robocasa.environments.kitchen.kitchen import *


class ToastBagel(Kitchen):
    """
    ToastBagel: composite task for Toasting Bread activity.

    Steps:
        1. Open the toaster oven door.
        2. Pick up the bagel from the plate on the counter.
        3. Place the bagel on the toaster oven rack.
        4. Close the oven door fully.
        5. Set the timer to start toasting.
    """

    # really small knob
    EXCLUDE_STYLES = [18, 49]

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
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the bagel from the plate, place it on the toaster oven rack, close the door, and set the timer."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_oven.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bagel",
                obj_groups="bagel",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.7, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        door_closed = self.toaster_oven.is_closed(self)
        gripper_obj_far = OU.gripper_obj_far(self, "bagel")

        if self.toaster_oven.has_multiple_rack_levels():
            bagel_on_lower = self.toaster_oven.check_rack_contact(
                self, "bagel", rack_level=0
            )
            bagel_on_upper = self.toaster_oven.check_rack_contact(
                self, "bagel", rack_level=1
            )
            bagel_in_toaster = bagel_on_lower or bagel_on_upper
        else:
            bagel_in_toaster = self.toaster_oven.check_rack_contact(self, "bagel")

        if bagel_in_toaster and door_closed:
            timer_val = self.toaster_oven.get_state(self)["time"]
            timer_set = timer_val >= 0.1
        else:
            return False

        return timer_set and gripper_obj_far
