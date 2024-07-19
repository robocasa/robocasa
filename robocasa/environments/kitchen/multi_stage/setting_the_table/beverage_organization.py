from robocasa.environments.kitchen.kitchen import *


class BeverageOrganization(Kitchen):

    def __init__(self, layout_ids=-4, *args, **kwargs):

        super().__init__( layout_ids=layout_ids, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        if "counter" in self.fixture_refs:
            self.counter = self.fixture_refs["counter"]
            self.dining_table = self.fixture_refs["dining_table"]
        else:

            self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)))
            self.counter = self.get_fixture(id=FixtureType.COUNTER)
            # do not want to sample the dining table or a counter with a builtin sink
            #TODO Change later!
            while self.counter == self.dining_table or "corner" in self.counter.name:
                self.counter = self.get_fixture(FixtureType.COUNTER)
            self.fixture_refs["counter"] = self.counter

        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Move the drinks to the dining counter"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        self.num_bev = self.rng.choice([2,3,4])
        for i in range(self.num_bev):
            cfgs.append(dict(
            name=f"obj_{i}",
            obj_groups="drink",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    top_size=(0.6, 0.4)
                    ),
                size=(0.6, 0.4),
                pos=(0, -1.0),
                ),
            ))

        return cfgs

    def _check_success(self):

        drinks_on_dining = all([OU.check_obj_fixture_contact(self, f"obj_{i}", self.dining_table) for i in range(self.num_bev)])
        return drinks_on_dining and OU.gripper_obj_far(self, "obj_0")