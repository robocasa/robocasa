from robocasa.environments.kitchen.kitchen import *


class StockCans(Kitchen):
    """
    Stock Cans: composite task for Restocking Supplies activity.

    Simulates the task of stocking dry goods in a cabinet.

    Steps:
        Pick a can from the cabinet and place it on the counter.
        Then, pick the other can from the counter and place it in the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Move 'old' can from cabinet to counter, 'new' can from counter to cabinet."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=1.0, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj_counter",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.30),
                    pos=(0.6, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.2, 0.30),
                    pos=(1, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs
    

    def _check_success(self):
        obj_counter_inside_cab = OU.obj_inside_of(self, "obj_counter", self.cab)
        obj_cab_on_counter = OU.check_obj_fixture_contact(self, "obj_cab", self.counter)
        gripper_obj_far = (OU.gripper_obj_far(self, "obj1") 
                           and OU.gripper_obj_far(self, "obj2")
        )

        return obj_counter_inside_cab and obj_cab_on_counter and gripper_obj_far
