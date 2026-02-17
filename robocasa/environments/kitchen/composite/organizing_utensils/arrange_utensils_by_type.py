from robocasa.environments.kitchen.kitchen import *


class ArrangeUtensilsByType(Kitchen):
    """
    Arrange Utensils By Type: composite task for Organizing Utensils activity.

    Simulates the task of organizing utensils by their material type.
    The wooden spoons needs to go in the cabinet and the metallic utensils go in the drawer.

    Steps:
        Pick up the wooden spoons from the counter and place it in the cabinet.
        Pick up the metallic utensils from the counter and place them in the drawer.
        Ensure both the cabinet and drawer are open for easy access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.8, 0.5))
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Organize the utensils by type. Pick up the wooden spoons from the counter "
            "and place them in the open cabinet. Then pick up the metallic utensils from the counter "
            "and place them in the open drawer."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)
        self.drawer.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="wooden_spoon",
                obj_groups="wooden_spoon",
                init_robot_here=True,
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="wooden_spoon_2",
                obj_groups="wooden_spoon",
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="wooden_spoon",
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="metallic_utensil",
                obj_groups=("spoon", "fork"),
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="wooden_spoon",
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="metallic_utensil_2",
                obj_groups=("spoon", "fork"),
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="wooden_spoon",
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        wooden_spoon_1_in_cabinet = OU.obj_inside_of(
            self, "wooden_spoon", self.cabinet
        ) and not OU.check_obj_fixture_contact(self, "wooden_spoon", self.counter)

        wooden_spoon_2_in_cabinet = OU.obj_inside_of(
            self, "wooden_spoon_2", self.cabinet
        ) and not OU.check_obj_fixture_contact(self, "wooden_spoon_2", self.counter)

        metallic_utensil_1_in_drawer = OU.obj_inside_of(
            self, "metallic_utensil", self.drawer
        ) and not OU.check_obj_fixture_contact(self, "metallic_utensil", self.counter)

        metallic_utensil_2_in_drawer = OU.obj_inside_of(
            self, "metallic_utensil_2", self.drawer
        ) and not OU.check_obj_fixture_contact(self, "metallic_utensil_2", self.counter)

        gripper_obj_far = (
            OU.gripper_obj_far(self, "wooden_spoon")
            and OU.gripper_obj_far(self, "wooden_spoon_2")
            and OU.gripper_obj_far(self, "metallic_utensil")
            and OU.gripper_obj_far(self, "metallic_utensil_2")
        )

        return (
            gripper_obj_far
            and wooden_spoon_1_in_cabinet
            and wooden_spoon_2_in_cabinet
            and metallic_utensil_1_in_drawer
            and metallic_utensil_2_in_drawer
        )
