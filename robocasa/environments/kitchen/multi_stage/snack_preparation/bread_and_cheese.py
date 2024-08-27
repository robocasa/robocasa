from robocasa.environments.kitchen.kitchen import *


class BreadAndCheese(Kitchen):
    """
    Bread And Cheese: composite task for Snack Preparation activity.

    Simulates the preparation of a bread and cheese snack.

    Steps:
        Pick the bread and cheese, place them on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = """pick the bread and cheese, place them on the cutting board,"""
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.50, 0.30),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups="cheese",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.3, 0.15),
                    pos=(0.0, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        food_on_cutting_board = OU.check_obj_in_receptacle(
            self, "obj", "container"
        ) and OU.check_obj_in_receptacle(self, "obj2", "container")
        return food_on_cutting_board and gripper_obj_far
