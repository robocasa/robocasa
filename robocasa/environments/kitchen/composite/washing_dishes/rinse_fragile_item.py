from robocasa.environments.kitchen.kitchen import *


class RinseFragileItem(Kitchen):
    """
    Rinse Fragile Item: composite task for Washing Dishes activity.

    Simulates the process of rinsing and securing a fragile item on a drying rack.

    Steps:
        1) Gently pick up the fragile item and navigate to the sink
        2) Rinse the item under running water
        3) Place the item on a drying rack
    """

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
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Rinse the teapot briefly and place it in the dish rack when done."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(self, rng=self.rng, mode="on")
        self.teapot_contact_time = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="teapot",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.6, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        in_water = self.sink.check_obj_under_water(self, "obj")

        if in_water:
            self.teapot_contact_time += 1

        gripper_far_teapot = OU.gripper_obj_far(self, obj_name="obj")
        teapot_in_rack = OU.obj_inside_of(self, "obj", self.dish_rack, th=0.02)

        return self.teapot_contact_time >= 100 and gripper_far_teapot and teapot_in_rack
