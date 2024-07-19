from robocasa.environments.kitchen.kitchen import *


class SetupJuicing(Kitchen):
    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        self.cab_id = cab_id
        self.num_fruits = self.rng.integers(low=2, high=4)
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.cab))
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"open the cabinet, pick all {self.num_fruits} fruits from the cabinet and place them on the counter"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(self.num_fruits):

            cfgs.append(dict(
                name=f"obj{i}",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.60, 0.40),
                    pos=(0, -1.0),
                ),
            ))

        cfgs.append(dict(
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
        ))
        cfgs.append(dict(
            name="distr_cab",
            obj_groups="all",
            placement=dict(
                fixture=self.cab,
                size=(1.0, 0.20),
                pos=(0.0, 1.0),
                offset=(0.0, 0.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        fruit_on_counter = all([OU.check_obj_fixture_contact(self, f"obj{i}", self.counter) for i in range(self.num_fruits)])
        return fruit_on_counter and OU.gripper_obj_far(self, "obj1")