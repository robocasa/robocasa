from robocasa.environments.kitchen.kitchen import *


class PlaceBreakfastItemsAway(Kitchen):
    """
    Place Breakfast Items Away: composite task for Tidying Cabinets and Drawers activity.

    Simulates the task of cleaning up after breakfast by placing breakfast items from the dining counter back into cabinet storage.

    Steps:
        1. Pick up breakfast items from the dining counter
        2. Place them in the cabinet for storage
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR, ref=self.dining_counter)
        )

        breakfast_options = ["cereal", "jam", "syrup_bottle", "honey_bottle"]
        if "refs" in self._ep_meta:
            self.breakfast_item1_type = self._ep_meta["refs"]["breakfast_item1_type"]
            self.breakfast_item2_type = self._ep_meta["refs"]["breakfast_item2_type"]
        else:
            self.breakfast_item1_type = self.rng.choice(breakfast_options)
            remaining_options = [
                item for item in breakfast_options if item != self.breakfast_item1_type
            ]
            self.breakfast_item2_type = self.rng.choice(remaining_options)

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        item1_name = self.get_obj_lang("breakfast_item1")
        item2_name = self.get_obj_lang("breakfast_item2")
        ep_meta["lang"] = (
            f"Pick up the {item1_name} and {item2_name} from the dining counter "
            f"and place them in the open cabinet to store away breakfast items."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["breakfast_item1_type"] = self.breakfast_item1_type
        ep_meta["refs"]["breakfast_item2_type"] = self.breakfast_item2_type
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="breakfast_item1",
                obj_groups=self.breakfast_item1_type,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.3, 0.25),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="breakfast_item2",
                obj_groups=self.breakfast_item2_type,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.6, 0.25),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_dining1",
                exclude_obj_groups=("cereal", "jam", "syrup_bottle", "honey_bottle"),
                graspable=True,
                fridgable=False,
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 0.5),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        item1_in_cabinet = OU.obj_inside_of(self, "breakfast_item1", self.cabinet)
        item2_in_cabinet = OU.obj_inside_of(self, "breakfast_item2", self.cabinet)

        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["breakfast_item1", "breakfast_item2"]
        )

        return item1_in_cabinet and item2_in_cabinet and gripper_far
