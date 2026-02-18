from robocasa.environments.kitchen.kitchen import *


class ServeWarmCroissant(Kitchen):
    """
    Serve Warm Croissant: composite task for Toasting Bread activity.

    Steps:
        1. Open the toaster oven door.
        2. Pull out the rack.
        3. Pick up the croissant and place it on the plate.
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
        if "rack_level" in self._ep_meta:
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Take out the warm croissant from the {rack_pos} rack of the toaster oven and place it on the plate."
        else:
            ep_meta[
                "lang"
            ] = "Take out the warm croissant from the toaster oven and place it on the plate."

        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="croissant",
                obj_groups="croissant",
                placement=dict(
                    fixture=self.toaster_oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(0.70, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.7, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        return OU.check_obj_in_receptacle(
            self, "croissant", "plate"
        ) and OU.gripper_obj_far(self, "croissant")
