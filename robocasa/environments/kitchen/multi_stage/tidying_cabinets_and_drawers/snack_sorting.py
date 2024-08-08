from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import *


class SnackSorting(ManipulateDrawer):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the bar in the bowl and close the drawer"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bar",
                obj_groups="bar",
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                    offset=(0, -self.drawer.size[1] * 0.55),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dist",
                obj_groups="all",
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, 1),
                    offset=(0, -self.drawer.size[1] * 0.55),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.60, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs
        return cfgs

    def _check_success(self):
        bars_in_bowl = OU.check_obj_in_receptacle(self, "bar", "bowl")

        door_closed = super()._check_success()

        return bars_in_bowl and door_closed
