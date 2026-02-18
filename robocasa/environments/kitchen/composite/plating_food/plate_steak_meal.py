from robocasa.environments.kitchen.kitchen import *


class PlateSteakMeal(Kitchen):
    """
    Plate Steak Meal: composite task for Plating Food activity.

    Simulates the process of plating a steak meal by moving specific food items
    from a tray to a plate while leaving other items on the tray.

    Steps:
        1. Pick up the steak from the tray and place it on the plate.
        2. Pick up the asparagus from the tray and place it on the plate.
        3. Leave the third random food item on the tray.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("aigen", "objaverse", "lightwheel"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "stool1" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
        else:
            registered_stool_ids = set()
            self.stool1 = None
            self.stool2 = None

            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                        fixture_id = id(fixture)
                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)
                            if self.stool1 is None:
                                self.stool1 = fixture
                            elif self.stool2 is None:
                                self.stool2 = fixture
                                break

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        random_food_lang = self.get_obj_lang("random_food")

        ep_meta["lang"] = (
            f"Create a steak meal by picking up the steak and asparagus from the tray and placing them on the plate. "
            f"Leave the {random_food_lang} on the tray."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray",
                obj_groups="tray",
                init_robot_here=True,
                object_scale=1.25,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="steak",
                obj_groups="steak",
                graspable=True,
                placement=dict(
                    object="tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="asparagus",
                obj_groups="asparagus",
                graspable=True,
                placement=dict(
                    object="tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="random_food",
                obj_groups=("fruit", "vegetable", "cooked_food"),
                exclude_obj_groups=("steak", "asparagus"),
                graspable=True,
                placement=dict(
                    object="tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        steak_on_plate = OU.check_obj_in_receptacle(self, "steak", "plate")
        asparagus_on_plate = OU.check_obj_in_receptacle(self, "asparagus", "plate")

        # Check that random food item is still on the tray
        random_food_on_tray = OU.check_obj_in_receptacle(self, "random_food", "tray")

        # Check that gripper is far from all objects
        gripper_far_steak = OU.gripper_obj_far(self, obj_name="steak")
        gripper_far_asparagus = OU.gripper_obj_far(self, obj_name="asparagus")
        gripper_far_random = OU.gripper_obj_far(self, obj_name="random_food")
        gripper_far = gripper_far_steak and gripper_far_asparagus and gripper_far_random

        return (
            steak_on_plate
            and asparagus_on_plate
            and random_food_on_tray
            and gripper_far
        )
