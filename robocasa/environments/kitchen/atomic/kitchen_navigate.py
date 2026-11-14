from robocasa.environments.kitchen.kitchen import *


class NavigateKitchen(Kitchen):
    """
    Class encapsulating the atomic navigate kitchen tasks.
    Involves navigating the robot to a target fixture.
    """

    def __init__(self, *args, **kwargs):
        # Enable the additional fixtures we want to use
        enable_fixtures = kwargs.get("enable_fixtures", [])
        enable_fixtures = list(enable_fixtures) + [
            "electric_kettle",
            "stand_mixer",
            "toaster_oven",
        ]
        kwargs["enable_fixtures"] = enable_fixtures
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
                "ToasterOven",
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
                "FridgeFrenchDoor",
                "FridgeSideBySide",
                "FridgeBottomFreezer",
                "Dishwasher",
                "ElectricKettle",
                "StandMixer",
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
                    "ToasterOven",
                    "Stove",
                    "Stovetop",
                    "OpenCabinet",
                    "Microwave",
                    "Sink",
                    "Hood",
                    "Oven",
                    "FridgeFrenchDoor",
                    "FridgeSideBySide",
                    "FridgeBottomFreezer",
                    "Dishwasher",
                    "ElectricKettle",
                    "StandMixer",
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
        ep_meta["lang"] = f"Navigate to the {self.target_fixture.nat_lang}."
        return ep_meta

    def _get_target_pose(self):
        """
        Get the robot base pose that counts as having reached the target fixture.

        `_setup_kitchen_references` computes this while `env.reset()` is building the
        scene, but a state restore replays a recorded scene against those same python
        fixtures -- `reset_from_xml_string` is only a soft reset -- so the value cached
        at reset time can describe a placement the simulator is not showing. The target
        is derived from the current fixture poses instead.

        Returns:
            tuple: (position, orientation) of the robot base
        """
        OU.sync_fixture_poses_from_sim(self)

        return EnvUtils.compute_robot_base_placement_pose(self, self.target_fixture)

    def _check_success(self):
        """
        Check if the navigation task is successful.
        This is done by checking if the robot is within a certain distance of the target fixture and the robot is facing the fixture.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        target_pos, target_ori = self._get_target_pose()
        robot_id = self.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])
        pos_check = np.linalg.norm(target_pos[:2] - base_pos[:2]) <= 0.20
        base_ori = T.mat2euler(
            np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        ori_check = np.cos(target_ori[2] - base_ori[2]) >= 0.98

        return pos_check and ori_check
