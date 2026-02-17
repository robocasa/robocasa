from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import *


class GatherCuttingTools(ManipulateDrawer):
    """
    Gather Cutting Tools: composite task for the chopping vegetables activity.
    Simulates the task of gathering cutting tools to prepare for chopping vegetables.
    Steps:
        Open the drawer on the specified side.
        Place the cutting tools (peeler and knife) on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.5, 0.5))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the drawer on the {self.drawer_side} and place the cutting tools on the cutting board."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="peeler",
                obj_groups="peeler",
                placement=dict(fixture=self.drawer, pos=(None, -0.7), size=(0.3, 0.3)),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                placement=dict(fixture=self.drawer, pos=(None, -0.7), size=(0.3, 0.3)),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="distr",
                placement=dict(
                    fixture=self.counter,
                    size=(0.6, 0.2),
                    sample_region_kwargs=dict(ref=self.drawer),
                    pos=(0, 1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer, top_size=(0.5, 0.5)),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        peeler_on_cutting_board = OU.check_obj_in_receptacle(
            self, "peeler", "cutting_board"
        )
        knife_on_cutting_board = OU.check_obj_in_receptacle(
            self, "knife", "cutting_board"
        )

        return (
            OU.gripper_obj_far(self, "cutting_board")
            and peeler_on_cutting_board
            and knife_on_cutting_board
        )
