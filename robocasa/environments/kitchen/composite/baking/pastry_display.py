import numpy as np

from robocasa.environments.kitchen.kitchen import *


class PastryDisplay(Kitchen):
    """
    Pastry Display: composite task for Baking activity.

    Simulates the task of displaying pastries.

    Steps:
        Place the pastries on the plates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.9, 0.4))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the pastries on the plates."
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
                name="receptacle1",
                obj_groups="plate",
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.90, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="plate",
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="receptacle1",
                    size=(0.90, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        # use offserts and to make it easier to initialuze pastry1 and pastry2
        cfgs.append(
            dict(
                name="pastry1",
                obj_groups="pastry",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="receptacle1",
                    size=(0.80, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pastry2",
                obj_groups="pastry",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="receptacle1",
                    size=(0.80, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_pastry1_far = OU.gripper_obj_far(self, obj_name="pastry1")
        gripper_pastry2_far = OU.gripper_obj_far(self, obj_name="pastry2")
        pastry1_in_receptacle1 = OU.check_obj_in_receptacle(
            self, "pastry1", "receptacle1"
        )
        pastry1_in_receptacle2 = OU.check_obj_in_receptacle(
            self, "pastry1", "receptacle2"
        )
        pastry2_in_receptacle1 = OU.check_obj_in_receptacle(
            self, "pastry2", "receptacle1"
        )
        pastry2_in_receptacle2 = OU.check_obj_in_receptacle(
            self, "pastry2", "receptacle2"
        )

        pastrys_placed = (pastry1_in_receptacle1 and pastry2_in_receptacle2) or (
            pastry1_in_receptacle2 and pastry2_in_receptacle1
        )

        return gripper_pastry1_far and gripper_pastry2_far and pastrys_placed
