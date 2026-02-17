from robocasa.environments.kitchen.kitchen import *


class PressChicken(Kitchen):
    """
    Press Chicken Drumstick for Crisp Finish: composite task for Frying Foods activity.

    Simulates the process of sizzling chicken drumstick while frying.

    Steps:
        1) Pick up the spatula on the plate by the stove.
        2) Press the spatula down on the chicken for 25 timesteps.
        3) Put the spatula back on the plate it was picked up from.
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
        ] = "Pick up the spatula from the plate and lightly press it against the chicken for a few seconds, then release."
        ep_meta["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        self.press_timer = 0
        self.press_done = False

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="chicken",
                obj_groups="chicken_drumstick",
                object_scale=1.15,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="spatula",
                obj_groups="spatula",
                object_scale=[1, 1, 3],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()
        spatula_obj = self.objects["spatula"]
        if "chicken" in self.objects:
            receptacle_obj = self.objects["chicken"]

        spatula_contact = self.check_contact(spatula_obj, receptacle_obj)

        if spatula_contact:
            self.press_timer += 1
        else:
            self.press_timer = 0

        if self.press_timer >= 25:
            self.press_done = True

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="chicken")
        if self.press_done and gripper_obj_far:
            return True

        return False
