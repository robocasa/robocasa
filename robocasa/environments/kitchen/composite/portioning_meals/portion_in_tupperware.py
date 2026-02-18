from robocasa.environments.kitchen.kitchen import *


class PortionInTupperware(Kitchen):
    """
    Portion In Tupperware: composite task for Portioning Meals activity.

    Simulates the task of portioning cooked food and vegetables from a plate into tupperware containers.
    The task involves taking 2 cooked food items and 2 vegetables from a plate on the dining counter
    and placing exactly 1 cooked food and 1 vegetable in each of the 2 tupperware containers.

    Steps:
        1. Take cooked food and vegetables from the plate on the dining counter
        2. Place exactly 1 cooked food and 1 vegetable in each tupperware container
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
        ] = "Place one cooked food and one vegetable from the large plate to each tupperware container."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware1",
                obj_groups="tupperware",
                object_scale=[1.75, 1.75, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware2",
                obj_groups="tupperware",
                object_scale=[1.75, 1.75, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.5,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(1.0, 0.45),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cooked_food1",
                obj_groups="cooked_food",
                exclude_obj_groups="kebab_skewer",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cooked_food2",
                obj_groups="cooked_food",
                exclude_obj_groups="kebab_skewer",
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="plate",
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
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cooked_food_in_tupperware1 = 0
        vegetable_in_tupperware1 = 0
        for food_name in ["cooked_food1", "cooked_food2"]:
            if OU.check_obj_in_receptacle(self, food_name, "tupperware1"):
                cooked_food_in_tupperware1 += 1
        for veg_name in ["vegetable1", "vegetable2"]:
            if OU.check_obj_in_receptacle(self, veg_name, "tupperware1"):
                vegetable_in_tupperware1 += 1

        cooked_food_in_tupperware2 = 0
        vegetable_in_tupperware2 = 0
        for food_name in ["cooked_food1", "cooked_food2"]:
            if OU.check_obj_in_receptacle(self, food_name, "tupperware2"):
                cooked_food_in_tupperware2 += 1
        for veg_name in ["vegetable1", "vegetable2"]:
            if OU.check_obj_in_receptacle(self, veg_name, "tupperware2"):
                vegetable_in_tupperware2 += 1

        gripper_far = True
        for obj_name in ["cooked_food1", "cooked_food2", "vegetable1", "vegetable2"]:
            if not OU.gripper_obj_far(self, obj_name):
                gripper_far = False
                break

        return (
            cooked_food_in_tupperware1 == 1
            and vegetable_in_tupperware1 == 1
            and cooked_food_in_tupperware2 == 1
            and vegetable_in_tupperware2 == 1
            and gripper_far
        )
