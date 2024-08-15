from robocasa.environments.kitchen.kitchen import *


class StockingBreakfastFoods(Kitchen):
    def __init__(
        self,
        cab_id1=FixtureType.CABINET_TOP,
        cab_id2=FixtureType.CABINET_TOP,
        *args,
        **kwargs,
    ):
        self.cab_id1 = cab_id1
        self.cab_id2 = cab_id2
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "cab2" in self.fixture_refs:
            self.cab1 = self.fixture_refs["cab1"]
            self.cab2 = self.fixture_refs["cab2"]
            self.counter = self.fixture_refs["counter"]
            self.counter2 = self.fixture_refs["counter2"]
        else:
            self.cab1 = self.register_fixture_ref("cab1", dict(id=self.cab_id1))

            while True:
                # would like different cabinets per notes on task redo
                self.cab2 = self.get_fixture(self.cab_id2)
                if self.cab2 != self.cab1:
                    break

            self.fixture_refs["cab2"] = self.cab2
            self.counter = self.register_fixture_ref(
                "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
            )
            self.counter2 = self.register_fixture_ref(
                "counter2", dict(id=FixtureType.COUNTER, ref=self.cab2)
            )
        self.init_robot_base_pos = self.cab1

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")

        # TODO: change for task
        ep_meta[
            "lang"
        ] = f"pick the {obj_name_1} and {obj_name_2} from the counter and place it in the closest cabinet to them."

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab1.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        self.cab2.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="packaged_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab1,
                    ),
                    size=(0.40, 0.20),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="packaged_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter2,
                    sample_region_kwargs=dict(
                        ref=self.cab2,
                    ),
                    size=(0.40, 0.20),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab1,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )

        # cfgs.append(dict(
        #     name="distr2_counter1",
        #     obj_groups="all",
        #     placement=dict(
        #         fixture=self.counter,
        #         sample_region_kwargs=dict(
        #             ref=self.cab1,
        #         ),
        #         size=(1.0, 0.30),
        #         pos=(0.0, 1.0),
        #         offset=(0.0, 0.05),
        #     ),
        # ))

        cfgs.append(
            dict(
                name="distr_cab1",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab1,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter2",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter2,
                    sample_region_kwargs=dict(
                        ref=self.cab2,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        # cfgs.append(dict(
        #     name="distr2_counter2",
        #     obj_groups="all",
        #     placement=dict(
        #         fixture=self.counter2,
        #         sample_region_kwargs=dict(
        #             ref=self.cab2,
        #         ),
        #         size=(1.0, 0.30),
        #         pos=(0.0, 1.0),
        #         offset=(0.0, 0.05),
        #     ),
        # ))

        cfgs.append(
            dict(
                name="distr_cab2",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab2,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_inside_cab1 = OU.obj_inside_of(self, "obj1", self.cab1)
        obj2_inside_cab2 = OU.obj_inside_of(self, "obj2", self.cab2)
        gripper_objs_far = OU.gripper_obj_far(self, "obj1") and OU.gripper_obj_far(
            self, "obj2"
        )

        return obj1_inside_cab1 and obj2_inside_cab2 and gripper_objs_far
