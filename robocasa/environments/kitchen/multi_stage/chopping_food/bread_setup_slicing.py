from robocasa.environments.kitchen.kitchen import *


class BreadSetupSlicing(Kitchen):
    """
    Bread Setup Slicing: composite task for Chopping Food activity.

    Simulates the task of setting up bread for slicing.

    Steps:
        Place all breads on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1.0, 0.4))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Place all breads on the cutting board."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
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
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(1, 0.4),
                    pos=(-0.6, -0.5),
                ),
            )
        )

        self.num_bread = self.rng.choice([1, 2, 3])
        for i in range(self.num_bread):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups="bread",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                        size=(1, 0.4),
                        pos=(0, -1.0),
                        offset=(i * 0.07, 0),
                        try_to_place_in="container",
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        bread_on_board = all(
            [
                OU.check_obj_in_receptacle(self, f"obj_{i}", "receptacle")
                for i in range(self.num_bread)
            ]
        )

        return bread_on_board and OU.gripper_obj_far(self, "obj_0")
