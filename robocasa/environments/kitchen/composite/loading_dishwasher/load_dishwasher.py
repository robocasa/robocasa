from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class LoadDishwasher(Kitchen):
    """
    LoadDishwasher: composite task for Washing Dishes activity.

    Stimulates loading dishes from the counter into the dishwasher.

    Steps:
        Take the dishes from the counter and place them on the top rack of the dishwasher.
        Close the dishwasher fully.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.dishwasher = self.register_fixture_ref(
            "dishwasher", dict(id=FixtureType.DISHWASHER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.dishwasher)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        dish0_lang = self.get_obj_lang("dish0")
        dish1_lang = self.get_obj_lang("dish1")
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                dish0_lang=dish0_lang, dish1_lang=dish1_lang
            )
        else:
            ep_meta["lang"] = (
                f"Pick up the {dish0_lang} and {dish1_lang} from the counter, place them in the dishwasher, "
                f"and close the dishwasher door."
            )
        return ep_meta

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        X_OFS = 0.25
        Y_OFS = -0.15

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

    def _setup_scene(self):
        super()._setup_scene()

        self.dishwasher.open_door(env=self)
        self.dishwasher.slide_rack(env=self, value=1.0)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="dish0",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.dishwasher,
                    ),
                    size=(0.35, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dish1",
                obj_groups="bowl",
                object_scale=0.5,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.dishwasher,
                    ),
                    size=(0.35, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Success if:
        1) All dishes are touching the top rack.
        2) The dishwasher door is fully closed.
        """
        is_dishwasher_closed = self.dishwasher.is_closed(self, th=0.05)

        dish_names = ["dish0", "dish1"]

        all_checks_passed = True
        for dish_name in dish_names:
            check1 = self.dishwasher.check_rack_contact(self, dish_name)
            check2 = self.dishwasher.check_rack_contact(self, dish_name)

            if not (check1 and check2):
                all_checks_passed = False
                break

        are_dishes_correctly_placed = all_checks_passed

        if are_dishes_correctly_placed and is_dishwasher_closed:
            return True
        else:
            return False
