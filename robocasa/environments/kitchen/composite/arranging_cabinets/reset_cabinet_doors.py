from robocasa.environments.kitchen.kitchen import *


class ResetCabinetDoors(Kitchen):
    """
    ResetCabinetDoors: composite task for Arranging Cabinets activity.

    Simulates the task of resetting cabinets doors.

    Steps:
        Close all open cabinet doors

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "cab1" in self.fixture_refs:
            self.cab = self.fixture_refs["cab1"]
            self.cab2 = self.fixture_refs["cab2"]
            self.cab3 = self.fixture_refs["cab3"]
        else:
            while True:
                self.cab = self.get_fixture(FixtureType.CABINET_WITH_DOOR)

                valid_cab_config_found = False
                for _ in range(100):  # 20 attempts
                    # sample until at least 2 different cabinets are selected
                    self.cab2 = self.get_fixture(FixtureType.CABINET_WITH_DOOR)
                    self.cab3 = self.get_fixture(FixtureType.CABINET_WITH_DOOR)
                    if (
                        self.cab2 != self.cab and self.cab3 != self.cab2
                    ):  # We only check for 2 different cabinets as there might only be two cabinets
                        valid_cab_config_found = True
                        break

                if valid_cab_config_found:
                    break

            self.fixture_refs["cab1"] = self.cab
            self.fixture_refs["cab2"] = self.cab2
            self.fixture_refs["cab3"] = self.cab3

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = "Close all open cabinet doors."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self, min=0.3, max=1.0)
        self.cab2.open_door(env=self, min=0.4, max=1.0)
        self.cab3.open_door(env=self, min=0.25, max=0.9)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                        # loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return (
            self.cab.is_closed(env=self)
            and self.cab2.is_closed(env=self)
            and self.cab3.is_closed(self)
        )
