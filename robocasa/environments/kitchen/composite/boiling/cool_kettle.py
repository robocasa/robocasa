from robocasa.environments.kitchen.kitchen import *


class CoolKettle(Kitchen):
    """
    Cool Kettle: composite task for Boiling Water activity.

    Simulates the task of waiting for boiled water to cool before handling.

    Steps:
        1) The kettle is already boiling on the stove (knob is on at the start).
        2) Wait 150 timesteps before picking it up.
        3) Move the kettle from the stove to the dining table.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.dining_table = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
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
        ep_meta[
            "lang"
        ] = "The kettle is done boiling. Turn off the stove, then wait for 150 timesteps before moving the kettle to the dining table."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cooldown_timer = 0
        self.cooling_done = False
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="kettle",
                obj_groups="kettle_non_electric",
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

        return cfgs

    def update_state(self):
        super().update_state()
        kettle_on_stove = OU.check_obj_fixture_contact(self, "kettle", self.stove)
        burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)

        if kettle_on_stove:
            if not burner_on:
                if not self.cooling_done:
                    self.cooldown_timer += 1
                    if self.cooldown_timer >= 150:
                        self.cooling_done = True

    def _check_success(self):
        kettle_on_counter = OU.check_obj_fixture_contact(
            self, "kettle", self.dining_table
        )
        gripper_obj_far = OU.gripper_obj_far(self, "kettle", th=0.15)

        return self.cooling_done and kettle_on_counter and gripper_obj_far
