from robocasa.environments.kitchen.kitchen import *


class MixedFruitPlatter(Kitchen):
    """
    Mixed Fruit Platter: composite task for Filling Serving Dishes activity.

    Simulates the task of serving fruit with cooked food.

    Steps:
        Pick up two fruits from the fridge and place one on each plate
        on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))

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

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool1),
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit1 = self.get_obj_lang("fruit1")
        fruit2 = self.get_obj_lang("fruit2")

        if fruit1 == fruit2:
            ep_meta["lang"] = (
                f"Pick up each {fruit1} from the fridge and place one {fruit1} "
                "on each plate on the dining counter."
            )
        else:
            ep_meta["lang"] = (
                f"Pick up the {fruit1} and {fruit2} from the fridge and place one fruit "
                "on each plate on the dining counter."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.2, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.2, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_fridge",
                obj_groups="food",
                exclude_obj_groups="fruit",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.30, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="cooked_food",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="cooked_food",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit1_on_plate1 = OU.check_obj_in_receptacle(
            self, "fruit1", "plate1_container"
        )
        fruit1_on_plate2 = OU.check_obj_in_receptacle(
            self, "fruit1", "plate2_container"
        )
        fruit2_on_plate1 = OU.check_obj_in_receptacle(
            self, "fruit2", "plate1_container"
        )
        fruit2_on_plate2 = OU.check_obj_in_receptacle(
            self, "fruit2", "plate2_container"
        )

        fruits_on_different_plates = (fruit1_on_plate1 and fruit2_on_plate2) or (
            fruit1_on_plate2 and fruit2_on_plate1
        )

        plates_on_table = OU.check_obj_fixture_contact(
            self, "plate1_container", self.dining_table
        ) and OU.check_obj_fixture_contact(self, "plate2_container", self.dining_table)

        gripper_far = OU.gripper_obj_far(self, "fruit1") and OU.gripper_obj_far(
            self, "fruit2"
        )

        return fruits_on_different_plates and plates_on_table and gripper_far
