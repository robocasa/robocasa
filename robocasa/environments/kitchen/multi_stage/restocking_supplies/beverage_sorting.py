from robocasa.environments.kitchen.kitchen import *


class BeverageSorting(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref(
            "cabinet1", dict(id=FixtureType.CABINET_TOP)
        )
        self.cab2 = self.register_fixture_ref(
            "cabinet2", dict(id=FixtureType.CABINET_TOP, ref=self.cab1)
        )
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
        self.cab1.set_door_state(min=0.85, max=0.9, env=self, rng=self.rng)
        self.cab2.set_door_state(min=0.85, max=0.9, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="alcohol1",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
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
                    fixture=self.counter,
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
                    fixture=self.counter,
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
                    fixture=self.counter,
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
