from robocasa.environments.kitchen.kitchen import *


class SortBreakfastIngredients(Kitchen):
    """
    Sort Breakfast Ingredients: composite task for Sorting Ingredients activity.

    Simulates the task of organizing breakfast items by placing jam next to bread
    and cereal next to milk on the dining counter.

    Steps:
        1. Pick up jam from the counter next to cabinet
        2. Place jam next to the plate of bread on dining counter
        3. Pick up cereal from the counter next to cabinet
        4. Place cereal next to the carton of milk on dining counter
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

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
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place the jam next to the plate of bread and the cereal next to the carton of milk on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.2, 0.2),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl_milk",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.6, 0.35),
                    pos=("ref", -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread_plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread",
                obj_groups=("bread", "sandwich_bread"),
                placement=dict(
                    object="bread_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cereal",
                obj_groups="cereal",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="jam",
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        jam_pos = self.sim.data.body_xpos[self.obj_body_id["jam"]]
        bread_plate_pos = self.sim.data.body_xpos[self.obj_body_id["bread_plate"]]
        jam_near_bread = np.linalg.norm(jam_pos[:2] - bread_plate_pos[:2]) <= 0.3

        cereal_pos = self.sim.data.body_xpos[self.obj_body_id["cereal"]]
        milk_pos = self.sim.data.body_xpos[self.obj_body_id["milk"]]
        cereal_near_milk = np.linalg.norm(cereal_pos[:2] - milk_pos[:2]) <= 0.3

        jam_on_dining_counter = OU.check_obj_any_counter_contact(self, "jam")
        cereal_on_dining_counter = OU.check_obj_any_counter_contact(self, "cereal")

        gripper_far_jam = OU.gripper_obj_far(self, "jam")
        gripper_far_cereal = OU.gripper_obj_far(self, "cereal")

        return (
            jam_near_bread
            and cereal_near_milk
            and jam_on_dining_counter
            and cereal_on_dining_counter
            and gripper_far_jam
            and gripper_far_cereal
        )
