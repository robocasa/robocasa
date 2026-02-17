from robocasa.environments.kitchen.kitchen import *


class SeparateFreezerRack(Kitchen):
    """
    Separate Freezer Rack: composite task for Managing Freezer Space activity.

    Simulates the task of organizing freezer items by placing meat on the second highest rack
    and vegetables on the highest rack of the freezer.

    Steps:
        1. Take the meat in its tupperware container
        2. Place the meat on the second highest rack of the freezer
        3. Take the vegetables in the other tupperware container
        4. Place the vegetables on the highest rack of the freezer
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
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        veg1_lang = self.get_obj_lang("vegetable1")
        veg2_lang = self.get_obj_lang("vegetable2")
        if self.use_novel_instructions:
            veg_lang = f"{veg1_lang} and {veg2_lang}"
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                meat_lang=meat_lang, veg_lang=veg_lang
            )
        elif veg1_lang == veg2_lang:
            ep_meta["lang"] = (
                f"Take the meat container that has the {meat_lang} and place it on the second highest rack of the freezer. "
                f"Then take the vegetable container that has the {veg1_lang}s and place it on the highest rack of the freezer."
            )
        else:
            ep_meta["lang"] = (
                f"Take the meat container that has the {meat_lang} and place it on the second highest rack of the freezer. "
                f"Then take the vegetable container that has the {veg1_lang} and {veg2_lang} and place it on the highest rack of the freezer."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat_tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.0, 0.4),
                    pos=(0.25, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    object="meat_tupperware",
                    size=(0.8, 0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg_tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="meat_tupperware",
                    size=(1.0, 0.4),
                    pos=(-0.25, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="veg_tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="veg_tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_in_tupperware = OU.check_obj_in_receptacle(self, "meat", "meat_tupperware")
        veg1_in_tupperware = OU.check_obj_in_receptacle(
            self, "vegetable1", "veg_tupperware"
        )
        veg2_in_tupperware = OU.check_obj_in_receptacle(
            self, "vegetable2", "veg_tupperware"
        )

        veg_tupperware_top_rack = self.fridge.check_rack_contact(
            self,
            "veg_tupperware",
            compartment="freezer",
            rack_index=-1,
        )
        meat_tupperware_second_rack = self.fridge.check_rack_contact(
            self,
            "meat_tupperware",
            compartment="freezer",
            rack_index=-2,
        )

        gripper_far_from_meat = OU.gripper_obj_far(self, obj_name="meat_tupperware")
        gripper_far_from_veg = OU.gripper_obj_far(self, obj_name="veg_tupperware")

        return (
            meat_in_tupperware
            and veg1_in_tupperware
            and veg2_in_tupperware
            and veg_tupperware_top_rack
            and meat_tupperware_second_rack
            and gripper_far_from_meat
            and gripper_far_from_veg
        )
