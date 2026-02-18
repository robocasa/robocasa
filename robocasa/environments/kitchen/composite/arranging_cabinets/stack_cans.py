from robocasa.environments.kitchen.kitchen import *


class StackCans(Kitchen):
    """
    Stack Cans: composite task for Restocking Supplies activity.

    Simulates the task of organizing cans in a cabinet.

    Steps:
        Stack 2 cans on top of the other 2 cans in the cabinet.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

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
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Stack two cans on top of the other two cans in the cabinet."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(self)

    def _get_obj_cfgs(self):
        object_scale = 1.0
        cfgs = []
        cfgs.append(
            dict(
                name="can1",
                obj_groups="canned_food",
                graspable=True,
                object_scale=object_scale,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.15),
                    pos=(0, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="can2",
                obj_groups="canned_food",
                graspable=True,
                object_scale=object_scale,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.15),
                    pos=(0, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="can3",
                obj_groups="canned_food",
                graspable=True,
                object_scale=object_scale,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.15),
                    pos=(0, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="can4",
                obj_groups="canned_food",
                graspable=True,
                object_scale=object_scale,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.15),
                    pos=(0, -0.9),
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

        return cfgs

    def _check_if_can_stacked(self, can1_name, can2_name):
        """
        Check if can1 is stacked on can2 or vice versa.
        """
        can1 = self.objects[can1_name]
        can2 = self.objects[can2_name]

        can1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[can1.name]])
        can2_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[can2.name]])

        dist = np.linalg.norm(can1_pos[0:1] - can2_pos[0:1])

        return dist < 0.02 and (
            (can1_pos[2] + 0.5 > can2_pos[2]) or (can2_pos[2] + 0.5 > can1_pos[2])
        )

    def _check_success(self):
        c1c2 = self._check_if_can_stacked("can1", "can2")
        c1c3 = self._check_if_can_stacked("can1", "can3")
        c1c4 = self._check_if_can_stacked("can1", "can4")

        c2c3 = self._check_if_can_stacked("can2", "can3")
        c2c4 = self._check_if_can_stacked("can2", "can4")

        c3c4 = self._check_if_can_stacked("can3", "can4")

        stacked = (c1c2 and c3c4) or (c1c3 and c2c4) or (c1c4 and c2c3)

        far_from_object1 = OU.gripper_obj_far(self, obj_name="can1")
        far_from_object2 = OU.gripper_obj_far(self, obj_name="can2")
        far_from_object3 = OU.gripper_obj_far(self, obj_name="can3")
        far_from_object4 = OU.gripper_obj_far(self, obj_name="can4")

        return (
            stacked
            and far_from_object1
            and far_from_object2
            and far_from_object3
            and far_from_object4
        )
