from robocasa.environments.kitchen.kitchen import *


class QuickThaw(Kitchen):
    """
    Quick Thaw: composite task for Defrosting Food activity.

    Simulates the task of defrosting meat on a stove.

    Steps:
        Pick the meat from the counter and place it in a pot on a burner.
        Then turn on the burner.

    Args:
        knob_id (str): The id of the knob who's burner the pot will be placed on.
            If "random", a random knob is chosen.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        # Pick a knob/burner on a stove and a counter close to it
        valid_knobs = [k for (k, v) in self.stove.knob_joints.items() if v is not None]
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.STOVE)
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Frozen meat rests on a plate on the counter. "
            "Retrieve the meat and place it in a pot on a burner. Then turn the burner on."
        )
        ep_meta["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.30),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        # place the pot on the specific burner we chose earlier
        cfgs.append(
            dict(
                name="container",
                obj_groups=("pot"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )
        return cfgs

    def _check_success(self):
        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]
        knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
        return (
            knob_on
            and OU.check_obj_in_receptacle(self, "meat", "container")
            and OU.gripper_obj_far(self, obj_name="meat")
        )
