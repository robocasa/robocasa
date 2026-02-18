from robocasa.environments.kitchen.kitchen import *


class PortionYogurt(Kitchen):
    """
    Portion Yogurt: composite task for Portioning Meals activity.

    Simulates the task of portioning yogurt from the dining counter into plates.
    The task involves taking 4 yogurt containers from the dining counter
    and placing exactly 2 yogurt containers on each of the 2 plates.

    Steps:
        1. Take yogurt containers from the dining counter
        2. Place 2 yogurt containers on each plate
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

        ep_meta["lang"] = "Place two yogurt containers on each plate."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.5, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.5, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt1",
                obj_groups="yogurt",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.5, 0.4),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt2",
                obj_groups="yogurt",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.5, 0.4),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt3",
                obj_groups="yogurt",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.5, 0.4),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="yogurt4",
                obj_groups="yogurt",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.5, 0.4),
                    pos=(0.25, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        yogurts_in_plate1 = 0
        for yogurt_name in ["yogurt1", "yogurt2", "yogurt3", "yogurt4"]:
            if OU.check_obj_in_receptacle(self, yogurt_name, "plate1"):
                yogurts_in_plate1 += 1

        yogurts_in_plate2 = 0
        for yogurt_name in ["yogurt1", "yogurt2", "yogurt3", "yogurt4"]:
            if OU.check_obj_in_receptacle(self, yogurt_name, "plate2"):
                yogurts_in_plate2 += 1

        gripper_far = True
        for yogurt_name in ["yogurt1", "yogurt2", "yogurt3", "yogurt4"]:
            if not OU.gripper_obj_far(self, yogurt_name):
                gripper_far = False
                break

        return yogurts_in_plate1 == 2 and yogurts_in_plate2 == 2 and gripper_far
