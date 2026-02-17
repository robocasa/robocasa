from robocasa.environments.kitchen.kitchen import *


class MatchCupAndDrink(Kitchen):
    """
    Match Cup and Drink: composite task for Serving Beverages activity.

    Simulates the task of matching drinks with appropriate cups on the dining counter.

    Steps:
        1. There is a wine glass and cup on the dining counter.
        2. There are wine and juice bottles on the counter next to the stove.
        3. Bring the wine bottle next to the wine glass.
        4. Bring the juice bottle next to the cup.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("aigen", "lightwheel", "objaverse"), *args, **kwargs
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
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool1),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.CABINET)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There is a wine glass and glass cup on the dining counter. "
            "Place the wine bottle next to the wine glass and the juice bottle next to the glass cup."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="wine_glass",
                obj_groups=("wine_glass"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool1),
                    size=(0.25, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups=("glass_cup"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool2),
                    size=(0.25, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        cfgs.append(
            dict(
                name="wine_bottle",
                obj_groups=("wine"),
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="juice_bottle",
                obj_groups=("juice"),
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="wine_bottle",
                    size=(0.8, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        wine_bottle_near_glass = False
        juice_bottle_near_cup = False

        wine_glass_pos = self.sim.data.body_xpos[self.obj_body_id["wine_glass"]][:2]
        wine_bottle_pos = self.sim.data.body_xpos[self.obj_body_id["wine_bottle"]][:2]
        wine_dist = np.linalg.norm(wine_glass_pos - wine_bottle_pos)
        wine_bottle_near_glass = wine_dist <= 0.25

        cup_pos = self.sim.data.body_xpos[self.obj_body_id["glass_cup"]][:2]
        juice_bottle_pos = self.sim.data.body_xpos[self.obj_body_id["juice_bottle"]][:2]
        juice_dist = np.linalg.norm(cup_pos - juice_bottle_pos)
        juice_bottle_near_cup = juice_dist <= 0.25

        gripper_far = OU.gripper_obj_far(
            self, obj_name="wine_bottle"
        ) and OU.gripper_obj_far(self, obj_name="juice_bottle")
        return wine_bottle_near_glass and juice_bottle_near_cup and gripper_far
