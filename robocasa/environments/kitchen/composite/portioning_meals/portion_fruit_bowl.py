from robocasa.environments.kitchen.kitchen import *


class PortionFruitBowl(Kitchen):
    """
    Portion Fruit Bowl: composite task for Portioning Meals activity.

    Simulates the task of portioning fruits from a plate into two bowls.
    The task involves taking 4 fruits from a plate on the dining counter
    and distributing them evenly into 2 bowls (2 fruits per bowl).

    Steps:
        1. Take fruits from the plate on the dining counter
        2. Place exactly 2 fruits in each of the 2 bowls
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + [22, 58]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        ep_meta[
            "lang"
        ] = "Take the fruits from the plate on the dining counter and place two fruits in each bowl."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.25,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit3",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit4",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruits_in_bowl1 = 0
        for fruit_name in ["fruit1", "fruit2", "fruit3", "fruit4"]:
            if OU.check_obj_in_receptacle(self, fruit_name, "bowl1"):
                fruits_in_bowl1 += 1

        fruits_in_bowl2 = 0
        for fruit_name in ["fruit1", "fruit2", "fruit3", "fruit4"]:
            if OU.check_obj_in_receptacle(self, fruit_name, "bowl2"):
                fruits_in_bowl2 += 1

        gripper_far = True
        for fruit_name in ["fruit1", "fruit2", "fruit3", "fruit4"]:
            if not OU.gripper_obj_far(self, fruit_name):
                gripper_far = False
                break

        return fruits_in_bowl1 == 2 and fruits_in_bowl2 == 2 and gripper_far
