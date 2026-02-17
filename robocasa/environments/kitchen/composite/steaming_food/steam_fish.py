from robocasa.environments.kitchen.kitchen import *


class SteamFish(Kitchen):
    """
    Steam Fish: composite task for Steaming Food activity.

    Simulates the task of steaming fish in a pan with soy sauce.

    Steps:
        1. Place the fish on the pan
        2. Turn the stove knob on

    Args:
        knob_id (str): The id of the knob whose burner the pan will be placed on.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.counter

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
        ] = f"Place the fish on the pan with soy sauce and turn the stove knob on to start steaming."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "pan", [0.28, 0.12, 0.05, 0.6])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                object_scale=[1, 1, 1.25],
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.03, 0.03),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="fish",
                obj_groups="fish",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        pan_loc = self.stove.check_obj_location_on_stove(
            env=self, obj_name="pan", threshold=0.15
        )
        pan_on_burner = pan_loc == self.knob

        fish_in_pan = OU.check_obj_in_receptacle(self, "fish", "pan")

        knobs_state = self.stove.get_knobs_state(env=self)
        burner_on = False
        if self.knob in knobs_state:
            burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)

        gripper_far = all(OU.gripper_obj_far(self, obj) for obj in ["fish", "pan"])

        return pan_on_burner and fish_in_pan and burner_on and gripper_far
