from robocasa.environments.kitchen.kitchen import *


class MoveToFreezerDrawer(Kitchen):
    """
    Move To Freezer Drawer: composite task for Managing Freezer Space activity.

    Simulates the task of moving fresh fruits or vegetables from the fridge shelf
    to the freezer drawer for freezing and long-term storage.

    Steps:
        1. Take the first fruit/vegetable from the fridge shelf
        2. Place it in the freezer drawer
        3. Take the second fruit/vegetable from the fridge shelf
        4. Place it in the freezer drawer
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS
    EXCLUDED_STYLES = [36]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj1_lang = self.get_obj_lang("obj1")
        obj2_lang = self.get_obj_lang("obj2")

        if obj1_lang == obj2_lang:
            objects_text = f"{obj1_lang}s"
        else:
            objects_text = f"{obj1_lang} and {obj2_lang}"

        ep_meta["lang"] = (
            f"Take the {objects_text} from the fridge shelf and "
            f"place them in the freezer drawer to begin freezing them."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="fridge")
        self.fridge.open_door(env=self, compartment="freezer")
        self.fridge.open_door(
            env=self, compartment="freezer", reg_type="drawer", drawer_rack_index=-1
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("fruit", "vegetable"),
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.15),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("fruit", "vegetable"),
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.15),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                exclude_obj_groups=("fruit", "vegetable"),
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.40, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_in_freezer_drawer = self.fridge.check_rack_contact(
            self,
            "obj1",
            compartment="freezer",
            reg_type=("drawer"),
            rack_index=-1,
        )
        obj2_in_freezer_drawer = self.fridge.check_rack_contact(
            self,
            "obj2",
            compartment="freezer",
            reg_type=("drawer"),
            rack_index=-1,
        )

        gripper_far = all(OU.gripper_obj_far(self, obj) for obj in ["obj1", "obj2"])

        return obj1_in_freezer_drawer and obj2_in_freezer_drawer and gripper_far
