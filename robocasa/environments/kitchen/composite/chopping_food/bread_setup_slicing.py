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
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.9, 0.4), full_depth_region=True),
        )
        self.init_robot_base_ref = self.counter
        if "refs" in self._ep_meta:
            self.num_bread = self._ep_meta["refs"]["num_bread"]
        else:
            self.num_bread = int(self.rng.choice([1, 2, 3]))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place all breads on the cutting board."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_bread"] = self.num_bread
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
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.9, 0.4),
                        full_depth_region=True,
                    ),
                    size=(1, 0.4),
                    pos=(0.0, -1.0),
                ),
            )
        )

        for i in range(self.num_bread):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups="bread",
                    graspable=True,
                    object_scale=0.75,
                    placement=dict(
                        fixture=self.counter,
                        reuse_region_from="receptacle",
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
