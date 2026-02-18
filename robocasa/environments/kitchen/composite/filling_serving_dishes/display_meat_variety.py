from robocasa.environments.kitchen.kitchen import *


class DisplayMeatVariety(Kitchen):
    """
    Display Meat Variety: composite task for Filling Serving Dishes activity.

    Simulates the task of retrieving different meat varieties from the fridge
    and displaying them on a serving tray on the dining counter.

    Steps:
        Retrieve chicken, shrimp, and beef pieces from the fridge and lay them out
        on a serving tray on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        chicken_name = self.get_obj_lang("chicken")
        shrimp_name = self.get_obj_lang("shrimp")
        beef_name = self.get_obj_lang("beef")

        ep_meta["lang"] = (
            f"Retrieve the {chicken_name}, {shrimp_name}, and {beef_name} from the fridge "
            "and place them on the serving tray on the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="chicken",
                obj_groups="chicken_drumstick",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="shrimp",
                obj_groups="shrimp",
                graspable=True,
                object_scale=[1, 1, 3],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="beef",
                obj_groups="steak",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="serving_tray",
                obj_groups="tray",
                graspable=False,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.6, 0.4),
                    pos=("ref", "ref"),
                    rotation=(np.pi / 2),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        chicken_on_tray = OU.check_obj_in_receptacle(self, "chicken", "serving_tray")
        shrimp_on_tray = OU.check_obj_in_receptacle(self, "shrimp", "serving_tray")
        beef_on_tray = OU.check_obj_in_receptacle(self, "beef", "serving_tray")

        tray_on_table = OU.check_obj_fixture_contact(
            self, "serving_tray", self.dining_table
        )

        gripper_far = (
            OU.gripper_obj_far(self, "chicken")
            and OU.gripper_obj_far(self, "shrimp")
            and OU.gripper_obj_far(self, "beef")
        )

        return (
            chicken_on_tray
            and shrimp_on_tray
            and beef_on_tray
            and tray_on_table
            and gripper_far
        )
