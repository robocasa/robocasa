from robocasa.environments.kitchen.kitchen import *


class WarmCroissant(Kitchen):
    """
    Warm Croissant: composite task for Reheating Food activity.

    Simulates the task of warming a croissant.

    Steps:
        Place the croissant on the pan and turn on the stove to warm the croissant.

    Args:
        knob_id (str): The id of the knob who's burner the pan will be placed on.
            If "random", a random knob is chosen.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        # Pick a knob/burner on a stove and a counter close to it
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

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.STOVE)
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the croissant and place it on the pan. Then turn on the stove to warm the croissant."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="croissant",
                obj_groups="croissant",
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.30),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    size=(0.02, 0.02),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)
        return (
            burner_on
            and OU.check_obj_in_receptacle(self, "croissant", "pan")
            and OU.gripper_obj_far(self, obj_name="croissant")
        )
