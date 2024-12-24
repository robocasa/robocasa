from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import *

# Inherit from ManipulateDrawer class since drawer is open to start
class SnackSorting(ManipulateDrawer):
    """
    Snack Sorting: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of placing snacks in the bowl.

    Steps:
        Place the bar in the bowl and close the drawer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the bar in the bowl and close the drawer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bar",
                obj_groups="bar",
                graspable=True,
                # have to make sure that the sampled object can fit inside the drawer hence the max_size being 0.1 in the z axis
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.20, 0.25),
                    # put object towards the front of the drawer
                    pos=(None, -0.75),
                    # offset to make sure the object is correctly placed since the drawer will be open to start
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
                    # offset to make sure the object is correctly placed since the drawer will be open to start
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
                    size=(0.15, 0.10),
                    offset=(0.0, 0.075),
                    pos=("ref", -1.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs
        return cfgs

    def _check_success(self):
        bars_in_bowl = OU.check_obj_in_receptacle(self, "bar", "bowl")

        # user super class to make sure that the drawer is closed
        door_closed = super()._check_success()

        return bars_in_bowl and door_closed
