from robocasa.environments.kitchen.kitchen import *


class SpicyMarinade(Kitchen):
    """
    Spicy Marinade: composite task for Mixing And Blending activity.

    Simulates the task of preparing a spicy marinade.

    Steps:
        Open the cabinet. Place the bowl and condiment on the counter. Then place
        the lime and garlic on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # need to fit the bowl and condiment in the cab, so use double door hinge
        self.cab = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the cabinet. Place the bowl and condiment on the counter. "
            "Then place the lime and garlic on the cutting board."
        )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.cab,
                    size=(0.6, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="lime",
                obj_groups="lime",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="garlic",
                obj_groups="garlic",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        objs_on_counter = OU.check_obj_fixture_contact(
            self, "bowl", self.counter
        ) and OU.check_obj_fixture_contact(self, "condiment", self.counter)
        objs_on_board = OU.check_obj_in_receptacle(
            self, "lime", "receptacle"
        ) and OU.check_obj_in_receptacle(self, "garlic", "receptacle")
        gripper_far = (
            OU.gripper_obj_far(self, "receptacle")
            and OU.gripper_obj_far(self, "bowl")
            and OU.gripper_obj_far(self, "condiment")
        )

        return objs_on_counter and objs_on_board and gripper_far
