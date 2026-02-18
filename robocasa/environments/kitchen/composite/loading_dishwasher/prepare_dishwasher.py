from robocasa.environments.kitchen.kitchen import *


class PrepareDishwasher(Kitchen):
    """
    PrepareDishwasher: composite task for Washing Dishes activity.

    Stimulates opening the dishwasher door and pulling out the top rack to prepare for loading.

    Steps:
        Open the dishwasher door.
        Pull out the top rack of the dishwasher.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.dishwasher = self.register_fixture_ref(
            "dishwasher", dict(id=FixtureType.DISHWASHER)
        )
        self.init_robot_base_ref = self.dishwasher

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the dishwasher door and pull out the top rack."
        return ep_meta

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        X_OFS = 0.275
        Y_OFS = -0.1

        x_ofs = (self.dishwasher.width / 2) + X_OFS
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(-x_ofs, Y_OFS)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(-x_ofs - TEST_OFS, Y_OFS)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not OU.check_fxtr_contact(
            self, test_pos_left
        ) and not OU.point_outside_scene(self, test_pos_left):
            # drawer is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the drawer
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(x_ofs, Y_OFS)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(x_ofs + TEST_OFS, Y_OFS)
        )

        if not OU.check_fxtr_contact(
            self, test_pos_right
        ) and not OU.point_outside_scene(self, test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            return False
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.drawer_side = side
        self.init_robot_base_pos_anchor = robot_base_pos
        self.init_robot_base_ori_anchor = robot_base_ori
        return True

    def _check_success(self):
        rack_frac_pulled = self.dishwasher.get_state(self)["rack"]
        rack_pulled = rack_frac_pulled >= 0.50

        if self.dishwasher.is_open(self, th=0.70) and rack_pulled:
            return True
        else:
            return False
