from robocasa.environments.kitchen.kitchen import *


class PlaceVegetablesEvenly(Kitchen):
    """
    Place Whole Vegetables Evenly: composite task for Sauteing Vegetables activity.

    Simulates the task of carefully arranging whole vegetables inside a pan on the stove,
    ensuring that they are evenly distributed for consistent heat exposure.

    Steps:
        1. Pick up whole vegetables from the counter.
        2. Place them inside the pan on the stove.
        3. Ensure that the vegetables are inside the pan and do not overlap.
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
        veg1_lang = self.get_obj_lang("veg1")
        veg2_lang = self.get_obj_lang("veg2")
        ep_meta["lang"] = (
            f"Pick up the {veg1_lang} and {veg2_lang} from the counter and place them inside the pan. "
            "Make sure they are evenly distributed and not overlapping each other."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        def veg_cfg(name, offset):
            return dict(
                name=name,
                obj_groups="vegetable",
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        top_size=(0.3, 0.3),
                        loc="left_right",
                    ),
                    size=(0.25, 0.25),
                    pos=("ref", offset),
                ),
            )

        return [
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
            ),
            veg_cfg("veg1", -0.4),
            veg_cfg("veg2", -0.6),
        ]

    def _check_success(self):
        vegs = ["veg1", "veg2"]

        # Check if both vegetables are in the pan
        all_in_pan = all(OU.check_obj_in_receptacle(self, veg, "pan") for veg in vegs)
        if not all_in_pan:
            return False

        # Get positions
        positions = [
            np.array(self.sim.data.body_xpos[self.obj_body_id[v]]) for v in vegs
        ]
        z_vals = [p[2] for p in positions]
        xy_vals = [p[:2] for p in positions]

        min_z_distance = 0.02
        min_xy_distance = 0.06

        # Check pairwise separation
        z_sep = abs(z_vals[0] - z_vals[1]) < min_z_distance
        xy_sep = np.linalg.norm(xy_vals[0] - xy_vals[1]) > min_xy_distance
        gripper_far = all(OU.gripper_obj_far(self, obj) for obj in vegs)

        return z_sep and xy_sep and gripper_far
