from robocasa.environments.kitchen.kitchen import *


class PlaceFoodInBowls(Kitchen):
    """
    Place Food In Bowls: composite task for Serving Food activity.

    Simulates the task of placing food in bowls.

    Steps:
        Pick up two bowls and place them on the counter.
        Then, pick up two food items and place them in the bowls.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowls are
            picked.
    """

    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        # use double door cabinet as default to have space for two bowls
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food1 = self.get_obj_lang("food1")
        food2 = self.get_obj_lang("food2")
        ep_meta[
            "lang"
        ] = f"Pick both bowls and place them on the counter. Then pick the {food1} and place it in one bowl and pick the {food2} and place it in the other bowl."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.4),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.4),
                    pos=(1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1",
                obj_groups="food_set1",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups="food_set1",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -0.5),
                    offset=(0.07, 0),
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
                    size=(0.50, 0.20),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_food1_far = OU.gripper_obj_far(self, obj_name="food1")
        gripper_food2_far = OU.gripper_obj_far(self, obj_name="food2")
        food1_in_receptacle1 = OU.check_obj_in_receptacle(self, "food1", "receptacle1")
        food1_in_receptacle2 = OU.check_obj_in_receptacle(self, "food1", "receptacle2")
        food2_in_receptacle1 = OU.check_obj_in_receptacle(self, "food2", "receptacle1")
        food2_in_receptacle2 = OU.check_obj_in_receptacle(self, "food2", "receptacle2")

        receptacles_on_counter = OU.check_obj_fixture_contact(
            self, "receptacle1", self.counter
        ) and OU.check_obj_fixture_contact(self, "receptacle2", self.counter)

        # make sure food are in different bowls
        food_in_bowls = (food1_in_receptacle1 and food2_in_receptacle2) or (
            food1_in_receptacle2 and food2_in_receptacle1
        )

        return (
            gripper_food1_far
            and gripper_food2_far
            and food_in_bowls
            and receptacles_on_counter
        )
