from robocasa.environments.kitchen.kitchen import *


class TiltPan(Kitchen):
    """
    Tilt the Pan to Move the Vegetable: composite task for Sauteing Vegetables activity.

    Simulates the task of gripping the pan's handle and tilting it lightly to slide the vegetables around.

    Steps:
        1. Grip the pan handle securely.
        2. Tilt the pan at an angle to move the item for 12 timesteps.
        3. Return the pan to its original position on the stove.
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
            f"Tilt the pan slightly to move the {veg_lang} inside the pan. "
            f"Place the pan back down on the stove once finished."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")

        self.success_time = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                    try_to_place_in="pan",
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()

        item_in_pan = OU.check_obj_in_receptacle(self, "obj", "obj_container")
        if item_in_pan:
            pan_quat = self.sim.data.body_xquat[self.obj_body_id["obj_container"]]
            pan_initial_quat = self.object_placements["obj_container"][1]
            dot_product = np.dot(pan_initial_quat, pan_quat)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            tilt_angle = 2.0 * np.arccos(dot_product)

            tilt_threshold = 0.3

            if tilt_angle > tilt_threshold:
                self.success_time += 1

    def _check_success(self):
        """
        - We want the user to tilt the pan enough so that vegetables stay inside,
        then put the pan back on the stove.

        Steps to check:
        1) Vegetables remain inside the pan.
        2) The user has tilted the pan at least once
            (angle from initial orientation > tilt_threshold).
        3) The user then returned the pan to the stove
            (angle from initial orientation < some small threshold),
            and the pan is on the stove.
        """

        item_in_pan = OU.check_obj_in_receptacle(self, "obj", "obj_container")

        if self.success_time >= 12:
            if (
                self.stove.check_obj_location_on_stove(
                    env=self, obj_name="obj_container", threshold=0.15
                )
                == self.knob
                and item_in_pan
            ):
                return True
            else:
                return False

        return False
