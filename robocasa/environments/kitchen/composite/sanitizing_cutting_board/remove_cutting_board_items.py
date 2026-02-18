from robocasa.environments.kitchen.kitchen import *


class RemoveCuttingBoardItems(Kitchen):
    """
    Clear Cutting Board: composite task for Sanitizing Cutting Board activity.

    Simulates the task of clearing items from a cutting board for cleaning the board.

    Steps:
        Remove two food items from the cutting board to prepare for cutting board for cleaning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_lang = self.get_obj_lang("vegetable")
        meat_lang = self.get_obj_lang("meat")
        ep_meta[
            "lang"
        ] = f"Remove the {vegetable_lang} and {meat_lang} from the cutting board to prepare it for cleaning."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="cutting_board",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    object="cutting_board",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        vegetable_off_board = not OU.check_obj_in_receptacle(
            self, "vegetable", "cutting_board"
        )

        meat_off_board = not OU.check_obj_in_receptacle(self, "meat", "cutting_board")

        gripper_far = OU.gripper_obj_far(self, "vegetable") and OU.gripper_obj_far(
            self, "meat"
        )

        return vegetable_off_board and meat_off_board and gripper_far
