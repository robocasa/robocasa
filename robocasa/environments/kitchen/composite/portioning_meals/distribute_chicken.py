from robocasa.environments.kitchen.kitchen import *


class DistributeChicken(Kitchen):
    """
    Distribute Chicken: composite task for Portioning Meals activity.

    Simulates the task of distributing cooked chicken drumsticks from a pan to plates.
    The task involves taking 2 chicken drumsticks from a pan and placing exactly
    1 drumstick on each of the 2 plates on the dining counter.

    Steps:
        1. Take chicken drumsticks from the pan
        2. Place exactly 1 drumstick on each plate
    """

    # layouts 22 and 58 have very high number of placement errors
    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + [22, 58]

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
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

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )
        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob

        ep_meta[
            "lang"
        ] = "Take the chicken drumsticks from the pan and place one on each plate on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                object_scale=1.25,
                init_robot_here=True,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.35),
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
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chicken_drumstick1",
                obj_groups="chicken_drumstick",
                graspable=True,
                placement=dict(
                    object="pan",
                    size=(0.9, 0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chicken_drumstick2",
                obj_groups="chicken_drumstick",
                graspable=True,
                placement=dict(
                    object="pan",
                    size=(0.9, 0.9),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        drumsticks_in_plate1 = 0
        for drumstick_name in ["chicken_drumstick1", "chicken_drumstick2"]:
            if OU.check_obj_in_receptacle(self, drumstick_name, "plate1"):
                drumsticks_in_plate1 += 1

        drumsticks_in_plate2 = 0
        for drumstick_name in ["chicken_drumstick1", "chicken_drumstick2"]:
            if OU.check_obj_in_receptacle(self, drumstick_name, "plate2"):
                drumsticks_in_plate2 += 1

        gripper_far = True
        for drumstick_name in ["chicken_drumstick1", "chicken_drumstick2"]:
            if not OU.gripper_obj_far(self, drumstick_name):
                gripper_far = False
                break

        return drumsticks_in_plate1 == 1 and drumsticks_in_plate2 == 1 and gripper_far
