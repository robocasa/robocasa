from robocasa.environments.kitchen.kitchen import *


class ReorganizeFrozenVegetables(Kitchen):
    """
    Reorganize Frozen Vegetables: composite task for Managing Freezer Space activity.

    Simulates the task of reorganizing frozen vegetables in the freezer by taking them
    out of their current locations and placing them in a tupperware container,
    and then putting the tupperware back in the freezer.

    Steps:
        1. Take the frozen vegetables from the freezer
        2. Place them in a tupperware container
        3. Put the tupperware containing the vegetables back in the freezer
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg1_lang = self.get_obj_lang("vegetable1")
        veg2_lang = self.get_obj_lang("vegetable2")
        ep_meta["lang"] = (
            f"Take the {veg1_lang} and {veg2_lang} out of the freezer and place them in the tupperware container on the counter. "
            "Then put the tupperware containing the vegetables back in the freezer."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.40, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg1_in_tupperware = OU.check_obj_in_receptacle(
            self, "vegetable1", "tupperware"
        )
        veg2_in_tupperware = OU.check_obj_in_receptacle(
            self, "vegetable2", "tupperware"
        )

        tupperware_in_freezer = self.fridge.check_rack_contact(
            self,
            "tupperware",
            compartment="freezer",
        )

        gripper_far_from_veg1 = OU.gripper_obj_far(self, obj_name="vegetable1")
        gripper_far_from_veg2 = OU.gripper_obj_far(self, obj_name="vegetable2")
        gripper_far_from_tupperware = OU.gripper_obj_far(self, obj_name="tupperware")

        return (
            veg1_in_tupperware
            and veg2_in_tupperware
            and tupperware_in_freezer
            and gripper_far_from_veg1
            and gripper_far_from_veg2
            and gripper_far_from_tupperware
        )
