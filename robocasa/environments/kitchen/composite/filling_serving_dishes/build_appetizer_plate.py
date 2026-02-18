from robocasa.environments.kitchen.kitchen import *


class BuildAppetizerPlate(Kitchen):
    """
    Build Appetizer Plate: composite task for Filling Serving Dishes activity.

    Simulates the task of creating an appetizer plate by selecting cheese, a meat item,
    and a vegetable from the fridge and arranging them on a plate.

    Steps:
        Take cheese, a meat item, and a vegetable from the fridge and place them
        on a plate to create an appetizer plate.
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
        cheese_name = self.get_obj_lang("cheese")
        meat_name = self.get_obj_lang("meat")
        vegetable_name = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"Take the {cheese_name}, {meat_name}, and {vegetable_name} from the fridge "
            f"and place them on the plate on the dining counter to create an appetizer plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
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
                name="vegetable",
                obj_groups="vegetable",
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
                name="meat",
                obj_groups="meat",
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
                name="appetizer_plate",
                obj_groups="plate",
                object_scale=1.25,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        cheese_on_plate = OU.check_obj_in_receptacle(self, "cheese", "appetizer_plate")
        meat_on_plate = OU.check_obj_in_receptacle(self, "meat", "appetizer_plate")
        vegetable_on_plate = OU.check_obj_in_receptacle(
            self, "vegetable", "appetizer_plate"
        )
        plate_on_table = OU.check_obj_fixture_contact(
            self, "appetizer_plate", self.dining_table
        )
        gripper_far = (
            OU.gripper_obj_far(self, "cheese")
            and OU.gripper_obj_far(self, "meat")
            and OU.gripper_obj_far(self, "vegetable")
        )
        return (
            cheese_on_plate
            and meat_on_plate
            and vegetable_on_plate
            and plate_on_table
            and gripper_far
        )
