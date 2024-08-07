from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import ManipulateDrawer


class DrawerUtensilSort(ManipulateDrawer):
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
