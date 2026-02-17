from robocasa.environments.kitchen.atomic.kitchen_drawer import *
from robocasa.environments.kitchen.kitchen import *


class PlaceStraw(ManipulateDrawer):
    """
    PlaceStraw: composite task for the making smoothies activity.
    Simulates the task of placing a straw in a cup.
    Steps:
        1. Open the drawer (left or right).
        2. Pick up the straw from the drawer.
        3. Put the straw in the cup on the counter.
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
        ep_meta[
            "lang"
        ] = f"Pick up the straw from the {self.drawer_side} drawer and put it inside the smoothie cup."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "cup", np.array([243, 201, 203, 255]) / 255.0)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.
        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="straw",
                obj_groups="straw",
                placement=dict(
                    fixture=self.drawer,
                    size=(0.2, 0.2),
                    pos=(None, 0),
                ),
                object_scale=[1.75, 1.0, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.15, 0.15),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):

        # need a more lenient threshold for bent straws which stick out
        straw_in_cup = OU.check_obj_in_receptacle(self, "straw", "cup", th=0.06)
        return straw_in_cup and OU.gripper_obj_far(self, "cup")
