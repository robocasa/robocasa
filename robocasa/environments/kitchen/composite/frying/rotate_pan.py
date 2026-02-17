from robocasa.environments.kitchen.kitchen import *


class RotatePan(Kitchen):
    """
    Rotate the Pan for Even Frying: composite task for Frying Foods activity.

    Simulates the task of rotating the pan slightly to distribute heat evenly.

    Steps:
        1) Ensure the pan is on the stove.
        2) Ensure meat/seafood is inside the pan.
        3) Rotate the pan.
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
        meat_obj = self.get_obj_lang("meat")
        ep_meta["lang"] = f"Rotate the pan to help distribute the heat more evenly."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        self.pan_rotated = False
        if hasattr(self, "_initial_yaw"):
            del self._initial_yaw

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
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

        return cfgs

    def _check_success(self):
        pan_on_stove = OU.check_obj_fixture_contact(self, "meat_container", self.stove)

        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "meat_container")

        pan_body_id = self.obj_body_id["meat_container"]
        pan_mat = self.sim.data.body_xmat[pan_body_id].reshape(3, 3)

        forward_x = pan_mat[0, 0]
        forward_y = pan_mat[1, 0]
        current_yaw = np.arctan2(forward_y, forward_x)

        if not hasattr(self, "_initial_yaw"):
            self._initial_yaw = current_yaw

        yaw_diff = np.abs(
            (current_yaw - self._initial_yaw + np.pi) % (2 * np.pi) - np.pi
        )
        rotated = yaw_diff > np.deg2rad(15)

        if rotated:
            self.pan_rotated = True

        gripper_pan_far = OU.gripper_obj_far(self, obj_name="meat_container")

        return pan_on_stove and meat_in_pan and self.pan_rotated and gripper_pan_far
