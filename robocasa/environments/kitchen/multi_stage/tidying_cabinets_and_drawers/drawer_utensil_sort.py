from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import ManipulateDrawer

# Inherit from ManipulateDrawer class since it involved opening drawers
class DrawerUtensilSort(ManipulateDrawer):
    """
    Drawer Utensil Sort: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of placing utensils in the drawer.

    Steps:
        Open the drawer and push the utensils inside it.

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
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.2, 0.2))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] += " and push the utensils inside it"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="utensil1",
                obj_groups="utensil",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups="utensil",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1.0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):

        # need to make sure utensils not on the counter. Since the top drawer (when closed) is so close to the counter,
        # its bounding box may overlap with the top of the counter. So when the utensil is on the counter it may be falsely identified as inside the drawer
        utensil1_inside_drawer = OU.obj_inside_of(
            self, "utensil1", self.drawer
        ) and not OU.check_obj_fixture_contact(self, "utensil1", self.counter)
        utensil2_inside_drawer = OU.obj_inside_of(
            self, "utensil2", self.drawer
        ) and not OU.check_obj_fixture_contact(self, "utensil2", self.counter)

        gripper_obj_far = OU.gripper_obj_far(
            self, obj_name="utensil1"
        ) and OU.gripper_obj_far(self, obj_name="utensil2")
        return utensil1_inside_drawer and utensil2_inside_drawer and gripper_obj_far
