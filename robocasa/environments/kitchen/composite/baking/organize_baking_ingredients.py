import numpy as np

from robocasa.environments.kitchen.kitchen import *


class OrganizeBakingIngredients(Kitchen):
    """
    Organize Baking Ingredients: composite task for Baking activity.

    Simulates the task of organizing baking ingredients.

    Steps:
        Place the eggs and milk near the bowl on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Gather the eggs and milk located in the vicinity of the sink area and place them right next to the bowl."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.6, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="egg1",
                obj_groups="egg",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.6, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="egg2",
                obj_groups="egg",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.6, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.6, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):

        th = 0.20

        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        egg1_pos = self.sim.data.body_xpos[self.obj_body_id["egg1"]]
        egg2_pos = self.sim.data.body_xpos[self.obj_body_id["egg2"]]
        milk_pos = self.sim.data.body_xpos[self.obj_body_id["milk"]]

        # check if the objects are near the bowl
        bowl_egg1_close = np.linalg.norm(bowl_pos - egg1_pos) < th
        bowl_egg2_close = np.linalg.norm(bowl_pos - egg2_pos) < th
        bowl_milk_close = np.linalg.norm(bowl_pos - milk_pos) < th

        gripper_objs_far = all(
            [
                OU.gripper_obj_far(self, obj_name, th=0.15)
                for obj_name in ["bowl", "egg1", "egg2", "milk"]
            ]
        )

        return (
            bowl_egg1_close and bowl_egg2_close and bowl_milk_close and gripper_objs_far
        )
