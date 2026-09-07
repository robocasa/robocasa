from robocasa.environments.kitchen.kitchen import *


class PortionPeaches(Kitchen):
    """
    Portion Peaches: composite task for Portioning Meals activity.

    Simulates the task of portioning peaches from a bowl into plates.
    The task involves taking 2 peaches from a bowl on the dining counter
    and placing exactly 1 peach on each of the 2 plates.

    Steps:
        1. Take the peaches from the bowl on the dining counter
        2. Place exactly 1 peach on each plate
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + [22, 58]

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
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )
        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Place one peach from the bowl on each plate."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    rotation=0,
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
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                object_scale=[1.5, 1.5, 1],
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 0.6),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="peach1",
                obj_groups="peach",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="peach2",
                obj_groups="peach",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        peaches_in_plate1 = 0
        for peach_name in ["peach1", "peach2"]:
            if OU.check_obj_in_receptacle(self, peach_name, "plate1"):
                peaches_in_plate1 += 1

        peaches_in_plate2 = 0
        for peach_name in ["peach1", "peach2"]:
            if OU.check_obj_in_receptacle(self, peach_name, "plate2"):
                peaches_in_plate2 += 1

        gripper_far = True
        for obj_name in ["peach1", "peach2"]:
            if not OU.gripper_obj_far(self, obj_name):
                gripper_far = False
                break

        return (
            peaches_in_plate1 == 1
            and peaches_in_plate2 == 1
            and gripper_far
        )
