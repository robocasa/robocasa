from robocasa.environments.kitchen.kitchen import *


class RestockPantry(Kitchen):
    """
    Restock Pantry: composite task for Restocking Supplies activity.

    Simulates the task of organizing cans when restocking them.

    Steps:
        Pick the cans from the counter and place them on the side of the cabinet
        that already has a can.
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
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the cans from the counter and place them in their designated side in the cabinet."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(min=1.0, max=1.0, env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1),
                ),
            )
        )

        # randomize the side of the cabinet that already has a can
        side = int(self.rng.choice([-1, 1]))

        cfgs.append(
            dict(
                name="cab_obj1",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.2, 0.30),
                    pos=(side, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj2",
                obj_groups="all",
                exclude_obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.2, 0.30),
                    pos=(side * -1, 0.3),
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

    def _close_to_cab_cans(self, obj_name, ratio=2):
        """
        Check if the object is closer to the cabinet cans than the other object

        Args:
            obj_name (str): name of the object

            ratio (float): ratio of the distance between the object and the cabinet cans to the distance
                between the object and the other object

        Returns:
            bool: True if the object is closer to the cabinet cans than the other object, False otherwise
        """
        obj = self.objects[obj_name]
        can = self.objects["cab_obj1"]
        other_obj = self.objects["cab_obj2"]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        can_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[can.name]])
        other_obj_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id[other_obj.name]]
        )

        can_dist = np.linalg.norm(obj_pos - can_pos)
        other_dist = np.linalg.norm(other_obj_pos - obj_pos)

        return can_dist * ratio < other_dist

    def _check_obj_in_cab(self, obj_name):
        return OU.obj_inside_of(self, obj_name, self.cab) and self._close_to_cab_cans(
            obj_name
        )

    def _check_success(self):
        obj1_inside_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(self, "obj2", self.cab)

        cans_close = self._close_to_cab_cans("obj1") and self._close_to_cab_cans("obj2")

        gripper_obj_far = OU.gripper_obj_far(self, "obj1") and OU.gripper_obj_far(
            self, "obj2"
        )

        return obj1_inside_cab and obj2_inside_cab and cans_close and gripper_obj_far
