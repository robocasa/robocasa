from robocasa.environments.kitchen.kitchen import *


class BeverageSorting(Kitchen):
    """
    Beverage Sorting: composite task for Restocking Supplies activity.

    Simulates the task of sorting beverages.

    Steps:
        Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the
        other.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "cabinet1" in self.fixture_refs:
            self.cab1 = self.fixture_refs["cabinet1"]
            self.cab2 = self.fixture_refs["cabinet2"]
            self.counter = self.fixture_refs["counter"]
        else:
            while True:
                self.cab1 = self.get_fixture(FixtureType.CABINET_TOP)

                valid_cab_config_found = False
                for _ in range(20):  # 20 attempts
                    # sample until 2 different cabinets are selected
                    self.cab2 = self.get_fixture(FixtureType.CABINET_TOP)
                    cab1_rot = self.cab1.rot % (2 * np.pi)
                    cab2_rot = self.cab2.rot % (2 * np.pi)
                    if self.cab2 != self.cab1 and np.abs(cab1_rot - cab2_rot) < 0.05:
                        valid_cab_config_found = True
                        break

                if valid_cab_config_found:
                    break

            self.fixture_refs["cabinet1"] = self.cab1
            self.fixture_refs["cabinet2"] = self.cab2
            self.counter = self.register_fixture_ref(
                "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5), ref=self.cab1)
            )

        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the other."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab1.set_door_state(min=0.85, max=0.9, env=self)
        self.cab2.set_door_state(min=0.85, max=0.9, env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="alcohol1",
                obj_groups="alcohol",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.5, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="alcohol2",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="non_alcohol1",
                obj_groups="drink",
                exclude_obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.5, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="non_alcohol2",
                obj_groups="drink",
                exclude_obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        gripper_far = True
        for obj_name in ["alcohol1", "alcohol2", "non_alcohol1", "non_alcohol2"]:
            gripper_far = gripper_far and OU.gripper_obj_far(self, obj_name=obj_name)
        if not gripper_far:
            return False

        # Two possible arrangements
        for (c1, c2) in [(self.cab1, self.cab2), (self.cab2, self.cab1)]:
            if OU.obj_inside_of(self, "alcohol1", c1) and OU.obj_inside_of(
                self, "alcohol2", c1
            ):
                if OU.obj_inside_of(self, "non_alcohol1", c2) and OU.obj_inside_of(
                    self, "non_alcohol2", c2
                ):
                    return True

        # return False otherwise
        return False
