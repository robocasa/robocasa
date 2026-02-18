from robocasa.environments.kitchen.kitchen import *


class SetBowlsForSoup(Kitchen):
    """
    Set Bowls For Soup: composite task for Setting The Table activity.

    Simulates the task of setting the table with bowls for soup.

    Steps:
        Move the bowls from the cabinet to the plates on the dining table.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowls are
            picked.
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS
    )

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))

        if "stool1" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
        else:
            registered_stool_ids = set()
            self.stool1 = None
            self.stool2 = None

            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                        fixture_id = id(fixture)
                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)
                            if self.stool1 is None:
                                self.stool1 = fixture
                            elif self.stool2 is None:
                                self.stool2 = fixture
                                break

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2

        self.counter_large = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool1),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Move the bowls from the cabinet to the plates on the dining table."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        # set both plates' ref as self.cab to put the plates in a similar area since dining table can be large
        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter_large,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter_large,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_bowl1_far = OU.gripper_obj_far(self, obj_name="bowl1")
        gripper_bowl2_far = OU.gripper_obj_far(self, obj_name="bowl2")
        bowl1_on_plate1 = OU.check_obj_in_receptacle(self, "bowl1", "plate1")
        bowl1_on_plate2 = OU.check_obj_in_receptacle(self, "bowl1", "plate2")
        bowl2_on_plate1 = OU.check_obj_in_receptacle(self, "bowl2", "plate1")
        bowl2_on_plate2 = OU.check_obj_in_receptacle(self, "bowl2", "plate2")

        bowls_set = (bowl1_on_plate1 and bowl2_on_plate2) or (
            bowl1_on_plate2 and bowl2_on_plate1
        )

        return gripper_bowl1_far and gripper_bowl2_far and bowls_set
