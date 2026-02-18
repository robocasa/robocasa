from robocasa.environments.kitchen.kitchen import *


class ButterOnPan(Kitchen):
    """
    Add Butter to the Pan: composite task for Sauteing Vegetables activity.

    Simulates the task of picking up butter from a plate on the counter and placing it into a pan on the stove.

    Steps:
        1. Identify the plate with butter on the counter.
        2. Pick up the butter.
        3. Place the butter into the pan on the stove.
        4. Turn on the burner for the pan on the stove.
    """

    def __init__(
        self,
        knob_id="random",
        obj_registries=("objaverse", "lightwheel", "aigen"),
        *args,
        **kwargs,
    ):
        self.knob_id = knob_id
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the butter
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
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
        ep_meta["lang"] = (
            f"Pick up the butter stick and place it in the pan. "
            f"Then turn on the burner for the pan."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
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
                name="butter",
                obj_groups="butter_stick",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.35, 0.35),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the task is successfully completed:
        1. The butter is in the pan.
        2. The pan is correctly placed on the stove.
        3. The burner under the pan is turned on.
        """
        butter_in_pan = OU.check_obj_in_receptacle(self, "butter", "pan")
        pot_loc = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pan", threshold=0.15
            )
            == self.knob
        )

        knobs_state = self.stove.get_knobs_state(env=self)
        burner_on = False
        if self.knob in knobs_state:
            burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)

        return butter_in_pan and pot_loc and burner_on
