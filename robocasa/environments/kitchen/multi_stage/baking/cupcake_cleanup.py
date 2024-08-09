import numpy as np

from robocasa.environments.kitchen.kitchen import *


class CupcakeCleanup(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Move the fresh-baked cupcake off the tray onto the counter, "
            "and place the bowl used for mixing into the sink."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cupcake",
                obj_groups="cupcake",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.4)
                    ),
                    size=(0.3, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.3, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_far = OU.gripper_obj_far(self, "cupcake") and OU.gripper_obj_far(
            self, "bowl"
        )
        bowl_in_sink = OU.obj_inside_of(self, "bowl", self.sink)
        cupcake_on_counter = OU.check_obj_fixture_contact(self, "cupcake", self.counter)

        return gripper_far and bowl_in_sink and cupcake_on_counter
