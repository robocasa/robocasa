from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import ManipulateDrawer

# Inherit from ManipulateDrawer class since it involved opening drawers
class DrawerUtensilSort(ManipulateDrawer):
    """
    Drawer Utensil Sort: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of placing utensils in the drawer.

    Steps:
        Open the drawer and place the utensils inside it.

    Args:
        drawer_id (int): Enum which serves as a unique identifier for different
            drawer types. Used to choose the drawer in which the utensils are
            placed. TOP_DRAWER specifies the uppermost drawer.
    """

    def __init__(self, drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        super().__init__(behavior="open", drawer_id=drawer_id, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.4, 0.2))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior} the {self.drawer_side} drawer and place the utensils inside it."
        ep_meta["lang"] = ep_meta["lang"][0].capitalize() + ep_meta["lang"][1:]
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="utensil1",
                obj_groups="utensil",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.3, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups="utensil",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.5, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        utensil1_inside_drawer = OU.obj_inside_of(
            self, "utensil1", self.drawer
        ) and not OU.check_obj_any_counter_contact(self, "utensil1")
        utensil2_inside_drawer = OU.obj_inside_of(
            self, "utensil2", self.drawer
        ) and not OU.check_obj_any_counter_contact(self, "utensil2")

        gripper_obj_far = OU.gripper_obj_far(
            self, obj_name="utensil1"
        ) and OU.gripper_obj_far(self, obj_name="utensil2")
        return utensil1_inside_drawer and utensil2_inside_drawer and gripper_obj_far
