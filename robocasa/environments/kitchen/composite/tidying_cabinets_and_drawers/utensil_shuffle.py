from robocasa.environments.kitchen.kitchen import *


class UtensilShuffle(Kitchen):
    """
    Utensil Shuffle: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of reorganizing the pantry by only placing the utensils in
    the drawer.

    Steps:
        Place the utensils in the drawer.
        Leave the other object(s) in the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.drawer)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick and place the utensils into the open drawer."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)
        self.drawer.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="utensil1",
                obj_groups="utensil",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups="utensil",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment_bottle",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0, 0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        utensil_in_drawer = (
            OU.obj_inside_of(self, "utensil1", self.drawer)
            and OU.obj_inside_of(self, "utensil2", self.drawer)
            and not OU.obj_inside_of(self, "condiment", self.drawer)
            and not OU.check_obj_any_counter_contact(self, "utensil1")
            and not OU.check_obj_any_counter_contact(self, "utensil2")
        )
        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["utensil1", "utensil2"]
        )
        return utensil_in_drawer and gripper_far
