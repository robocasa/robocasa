from robocasa.environments.kitchen.kitchen import *


class ManipulateDrawer(Kitchen):
    """
    Class encapsulating the atomic manipulate drawer tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            drawer manipulation behavior for the task.

        drawer_id (str): The drawer fixture id to manipulate
    """

    def __init__(
        self,
        behavior="open",
        drawer_id=FixtureType.TOP_DRAWER,
        robot_spawn_deviation_pos_x=0.05,
        *args,
        **kwargs,
    ):
        # for open and close tasks the robot will next to the drawer.
        # since for open tasks all drawers will be close it is ambigious which drawer to open,
        # so we specify a which side the drawer is on to open. The side is determined in _load_model
        self.robot_side = ""
        self.drawer_id = drawer_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(
            robot_spawn_deviation_pos_x=robot_spawn_deviation_pos_x, *args, **kwargs
        )

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        x_ofs = (self.drawer.width / 2) + 0.20
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs - TEST_OFS, -0.10)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not self.check_fxtr_contact(test_pos_left) and not self._point_outside_scene(
            test_pos_left
        ):
            # drawer is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the drawer
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs + TEST_OFS, -0.10)
        )

        if not self.check_fxtr_contact(
            test_pos_right
        ) and not self._point_outside_scene(test_pos_right):
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
        """
        Reset the environment internal state for the drawer tasks.
        This includes setting the drawer state based on the behavior
        """
        if self.behavior == "open":
            self.drawer.close_door(env=self)
        elif self.behavior == "close":
            self.drawer.open_door(env=self)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._setup_scene()

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the drawer tasks
        """
        super()._setup_kitchen_references()
        valid_drawer = False
        for i in range(7):
            self.drawer = self.get_fixture(id=self.drawer_id)
            if self._place_robot():
                valid_drawer = True
                break
            if macros.VERBOSE:
                print(f"Attempt {i} to place robot failed")
        if not valid_drawer:
            if macros.VERBOSE:
                print("Could not place robot. Trying again with self._load_model()")
            self._destroy_sim()
            self._load_model()
            return

        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer))
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        """
        Get the episode metadata for the drawer tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior.capitalize()} the {self.drawer_side} drawer."
        return ep_meta

    def check_fxtr_contact(self, pos):
        """
        Check if the point is in contact with any fixture

        Args:
            pos (tuple): The position of the point to check

        Returns:
            bool: True if the point is in contact with any fixture, False otherwise
        """
        fxtrs = [
            fxtr
            for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, SingleCabinet)
            or isinstance(fxtr, HingeCabinet)
            or isinstance(fxtr, Fridge)
            or (isinstance(fxtr, Wall) and not isinstance(fxtr, Floor))
        ]

        for fxtr in fxtrs:
            # get bounds of fixture
            if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                return True
        return False

    def _point_outside_scene(self, pos):
        walls = [
            fxtr for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Floor)
        ]
        return not any(
            [
                OU.point_in_fixture(point=pos, fixture=wall, only_2d=True)
                for wall in walls
            ]
        )

    def _check_success(self):
        """
        Check if the drawer manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.95:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success


class OpenDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic open drawer task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups=("tool", "utensil"),
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class CloseDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic close drawer task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups=("tool", "utensil"),
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class SlideDishwasherRack(Kitchen):
    """
    Class encapsulating sliding dishwasher rack in or out atomic task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dishwasher = self.register_fixture_ref(
            "dishwasher", dict(id=FixtureType.DISHWASHER)
        )
        if "should_pull" in self._ep_meta:
            self.should_pull = self._ep_meta["should_pull"]
        else:
            self.should_pull = self.rng.random() > 0.5

        self.init_robot_base_ref = self.dishwasher

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"
        ep_meta["lang"] = f"Fully slide the top dishwasher rack {direction}."
        ep_meta["should_pull"] = self.should_pull
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.dishwasher.open_door(self)

        if not self.should_pull:
            self.dishwasher.slide_rack(self)
        else:
            self.dishwasher.slide_rack(self, value=0.5)

    def _check_success(self):
        current_pos = self.dishwasher.get_state(self)["rack"]

        if self.should_pull:
            return current_pos >= 0.95
        else:
            return current_pos <= 0.05


class OpenFridgeDrawer(Kitchen):
    """
    Class encapsulating the atomic open fridge drawer task to pull the fridge drawer out.
    """

    # No drawer in side-by-side fridge in these styles
    EXCLUDE_STYLES = [23, 24, 25, 27, 28, 37, 38, 40, 44, 47, 56]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Fully open the fridge drawer."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, min=1.0, max=1.0)
        self.fridge.open_door(
            self, min=0.1, max=0.3, reg_type="drawer", drawer_rack_index=-1
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="distr_drawer",
                fridgable=True,
                graspable=True,
                obj_groups=("vegetable", "fruit"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("drawer"),
                        rack_index=-1,
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_shelf",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("shelf"),
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return self.fridge.is_open(
            self, compartment="fridge", reg_type="drawer", drawer_rack_index=-1, th=0.70
        )


class CloseFridgeDrawer(Kitchen):
    """
    Class encapsulating the atomic close fridge drawer task to push the fridge drawer in.
    """

    # No drawer in side-by-side fridge in these styles
    EXCLUDE_STYLES = [23, 24, 25, 27, 28, 37, 38, 40, 44, 47, 56]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Fully close the fridge drawer."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, min=1.0, max=1.0)
        self.fridge.open_door(
            self, min=0.8, max=1.0, reg_type="drawer", drawer_rack_index=-1
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="distr_drawer",
                fridgable=True,
                graspable=True,
                obj_groups=("vegetable", "fruit"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("drawer"),
                        rack_index=-1,
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_shelf",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("shelf"),
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return self.fridge.is_closed(
            self, compartment="fridge", reg_type="drawer", drawer_rack_index=-1, th=0.05
        )
