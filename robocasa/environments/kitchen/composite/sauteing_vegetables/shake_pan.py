from robocasa.environments.kitchen.kitchen import *


class ShakePan(Kitchen):
    """
    Shake the Pan to Move the Vegetable: composite task for Sauteing Vegetables activity.

    Simulates the task of gripping the pan's handle and shaking it lightly to roll and move the vegetable around.

    Steps:
        1. Grip the pan handle securely.
        2. Shake the pan to move the vegetable for 25 timesteps.
        3. Put pan back down on stove.
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
        veg_lang = self.get_obj_lang("obj")
        ep_meta["lang"] = (
            f"Grip the pan handle, lift and shake it lightly to move the {veg_lang} inside the pan. "
            "Place the pan back down on the stove once finished."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_time = 0
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")

    def _get_obj_cfgs(self):
        return [
            dict(
                name="obj",
                obj_groups="vegetable",
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                    try_to_place_in="pan",
                ),
            )
        ]

    def update_state(self):
        super().update_state()
        # Check if the vegetable is being shaken and update success time accordingly
        veg_in_pan = OU.check_obj_in_receptacle(self, "obj", "obj_container")
        if veg_in_pan:
            pan_pos = np.array(
                self.sim.data.body_xpos[self.obj_body_id["obj_container"]]
            )
            prev_pan_pos = getattr(self, "prev_pan_pos", pan_pos)
            movement_value = np.linalg.norm(pan_pos - prev_pan_pos)
            self.prev_pan_pos = pan_pos

            if movement_value > 0.001:
                self.success_time += 1

    def _check_success(self):
        veg_in_pan = OU.check_obj_in_receptacle(self, "obj", "obj_container")

        if self.success_time >= 25:
            on_stove = (
                self.stove.check_obj_location_on_stove(
                    env=self, obj_name="obj_container", threshold=0.15
                )
                == self.knob
            )
            return veg_in_pan and on_stove
        return False
