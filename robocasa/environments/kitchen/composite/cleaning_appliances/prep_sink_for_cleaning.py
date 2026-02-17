from robocasa.environments.kitchen.kitchen import *


class PrepSinkForCleaning(Kitchen):
    """
    Prep Sink for Cleaning: composite task for Cleaning Appliances activity.

    Simulates the task of clearing the sink area by moving objects away to the cabinet and fridge
    before cleaning the faucet.

    Steps:
        1. Move the cup from near the sink to the open cabinet.
        2. Move the bowl with food item to the fridge to clear the area for faucet cleaning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_item_name = self.get_obj_lang("food_item")
        cup_name = self.get_obj_lang("cup")
        ep_meta["lang"] = (
            f"Move the {cup_name} from next to the sink to the open cabinet. "
            f"Then, move the bowl with the {food_item_name} to the fridge to clear the area for faucet cleaning."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(self)
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="food_item",
                obj_groups=("meat", "fruit", "vegetable"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )
        cfgs.append(
            dict(
                name="cup",
                obj_groups=("cup", "mug", "glass_cup"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.75, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="cleaner_distractor",
                obj_groups=("cleaner"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(1.0, 0.4),
                    pos=(0, 1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        food_item_in_fridge = self.fridge.check_rack_contact(
            self, "food_item_container"
        )
        cup_in_cabinet = OU.check_obj_fixture_contact(self, "cup", self.cabinet)

        gripper_far_food_item = OU.gripper_obj_far(self, "food_item_container")
        gripper_far_cup = OU.gripper_obj_far(self, "cup")

        return (
            food_item_in_fridge
            and cup_in_cabinet
            and gripper_far_food_item
            and gripper_far_cup
        )
