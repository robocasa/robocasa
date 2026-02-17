from robocasa.environments.kitchen.kitchen import *


class ClearFreezer(Kitchen):
    """
    Clear Freezer: composite task for Managing Freezer Space activity.

    Simulates the task of clearing the freezer by taking out all items and placing them in a bowl.

    Steps:
        1. Take all 3 items from the freezer
        2. Place them in the bowl on the counter
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
        item1_lang = self.get_obj_lang("item1")
        item2_lang = self.get_obj_lang("item2")
        item3_lang = self.get_obj_lang("item3")
        ep_meta["lang"] = (
            f"Take the {item1_lang}, {item2_lang}, and {item3_lang} from the freezer "
            "and place them in the bowl on the counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="item1",
                obj_groups=["vegetable", "fruit", "meat"],
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.40, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item2",
                obj_groups=["vegetable", "fruit", "meat"],
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.40, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item3",
                obj_groups=["vegetable", "fruit", "meat"],
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.40, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.5, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        item1_in_bowl = OU.check_obj_in_receptacle(self, "item1", "bowl")
        item2_in_bowl = OU.check_obj_in_receptacle(self, "item2", "bowl")
        item3_in_bowl = OU.check_obj_in_receptacle(self, "item3", "bowl")
        bowl_on_counter = OU.check_obj_fixture_contact(self, "bowl", self.counter)

        gripper_far_from_item1 = OU.gripper_obj_far(self, obj_name="item1")
        gripper_far_from_item2 = OU.gripper_obj_far(self, obj_name="item2")
        gripper_far_from_item3 = OU.gripper_obj_far(self, obj_name="item3")

        return (
            item1_in_bowl
            and item2_in_bowl
            and item3_in_bowl
            and bowl_on_counter
            and gripper_far_from_item1
            and gripper_far_from_item2
            and gripper_far_from_item3
        )
