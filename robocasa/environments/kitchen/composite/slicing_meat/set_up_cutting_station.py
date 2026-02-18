from robocasa.environments.kitchen.kitchen import *


class SetUpCuttingStation(Kitchen):
    """
    Set Up Cutting Station: composite task for Slicing Meat activity.
    Simulates the task of setting up a cutting station.

    Steps:
        1. Move the meat to the cutting board.
        2. Pick the knife to the cutting board from the drawer.

    """

    # I found that sliding the cutting board off the counter slightly to pick it up to be easiest.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER)
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.counter)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = f"Pick up the knife from the drawer and place it on the cutting board. Then place the meat from the plate to the cutting board."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        self.drawer.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                object_scale=[1.25, 1.25, 2.5],
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                    offset=(0, 0.10),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):

        meat_on_board = OU.check_obj_in_receptacle(self, "meat", "receptacle")

        knife_on_board = OU.check_obj_in_receptacle(self, "knife", "receptacle")

        gripper_far = OU.gripper_obj_far(self, "knife") and OU.gripper_obj_far(
            self, "receptacle"
        )

        return meat_on_board and knife_on_board and gripper_far
