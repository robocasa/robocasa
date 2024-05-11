import numpy as np
from robocasa.environments.kitchen.kitchen import *


class OrganizeBakingIngredients(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink))
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"pick place the eggs and milk to near the bowl"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="bowl",
            obj_groups="bowl",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.40, 0.40),
                pos=(0.0, -1.0),
            ),
        ))

        cfgs.append(dict(
            name="egg1",
            obj_groups="egg",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.3, 0.3),
                pos=(-1.0, -0.4),
            ),
        ))

        cfgs.append(dict(
            name="egg2",
            obj_groups="egg",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.3, 0.3),
                pos=(-1.0, -0.4),
                offset=(0.2, 0.0)
            ),
        ))

        cfgs.append(dict(
            name="milk",
            obj_groups="milk",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.5, 0.5),
                pos=(1.0, -1.0),
            ),
        ))

        return cfgs

    def _check_success(self):

        th = 0.2
        
        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        egg1_pos = self.sim.data.body_xpos[self.obj_body_id["egg1"]]
        egg2_pos = self.sim.data.body_xpos[self.obj_body_id["egg2"]]
        milk_pos = self.sim.data.body_xpos[self.obj_body_id["milk"]]

        bowl_egg1_close = np.linalg.norm(bowl_pos - egg1_pos) < th
        bowl_egg2_close = np.linalg.norm(bowl_pos - egg2_pos) < th
        bowl_milk_close = np.linalg.norm(bowl_pos - milk_pos) < th

        return bowl_egg1_close and bowl_egg2_close and bowl_milk_close