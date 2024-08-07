from robocasa.environments.kitchen.kitchen import *


class ClearingTheCuttingBoard(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Clear the non-vegetable object off the cutting board and place the "
            "vegetables onto it."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="non_vegetable",
                graspable=True,
                obj_groups="food",
                exclude_obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    size=(0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    pos=(0, -0.3),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
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
                name="vegetable2",
                obj_groups="vegetable",
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
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            self, "vegetable1", "non_vegetable_container"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            self, "vegetable2", "non_vegetable_container"
        )
        cutting_board_cleared = not OU.check_obj_in_receptacle(
            self, "non_vegetable", "non_vegetable_container"
        )
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="non_vegetable_container")

        return (
            vegetable1_cutting_board_contact
            and vegetable2_cutting_board_contact
            and gripper_obj_far
            and cutting_board_cleared
        )
