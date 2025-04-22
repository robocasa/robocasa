from robocasa.environments.kitchen.kitchen import *


class NavigateKitchen(Kitchen):
    """
    Class encapsulating the atomic navigate kitchen tasks.
    Involves navigating the robot to a target fixture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the navigate kitchen tasks.
        If not already chosen, selects a random start and destination fixture for the robot to navigate from/to.
        """
        super()._setup_kitchen_references()
        if "src_fixture" in self.fixture_refs:
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            # choose a valid random start and destination fixture
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                "CoffeeMachine",
                "Toaster",
                "Stove",
                "Stovetop",
                "SingleCabinet",
                "HingeCabinet",
                "OpenCabinet",
                "Drawer",
                "Microwave",
                "Sink",
                "Hood",
                "Oven",
                "Fridge",
                "Dishwasher",
            ]
            # keep choosing src fixture until it is a valid fixture
            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break

            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls
                for cls in fxtr_classes
                if fxtr_classes.count(cls) == 1
                and cls
                in [
                    "CoffeeMachine",
                    "Toaster",
                    "Stove",
                    "Stovetop",
                    "OpenCabinet",
                    "Microwave",
                    "Sink",
                    "Hood",
                    "Oven",
                    "Fridge",
                    "Dishwasher",
                ]
            ]

            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if (
                    self.target_fixture == self.src_fixture
                    or fxtr_class not in valid_target_fxtr_classes
                ):
                    continue
                if fxtr_class == "Accessory":
                    continue
                # don't sample closeby fixtures
                if (
                    OU.fixture_pairwise_dist(self.src_fixture, self.target_fixture)
                    <= 1.0
                ):
                    continue
                break

            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture

        self.target_pos, self.target_ori = EnvUtils.compute_robot_base_placement_pose(
            self, self.target_fixture
        )

        self.init_robot_base_ref = self.src_fixture

    def get_ep_meta(self):
        """
        Get the episode metadata for the navigate kitchen tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"navigate to the {self.target_fixture.nat_lang}"
        return ep_meta

    def _check_success(self):
        """
        Check if the navigation task is successful.
        This is done by checking if the robot is within a certain distance of the target fixture and the robot is facing the fixture.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        robot_id = self.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])
        pos_check = np.linalg.norm(self.target_pos[:2] - base_pos[:2]) <= 0.20
        base_ori = T.mat2euler(
            np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        ori_check = np.cos(self.target_ori[2] - base_ori[2]) >= 0.98

        return pos_check and ori_check
