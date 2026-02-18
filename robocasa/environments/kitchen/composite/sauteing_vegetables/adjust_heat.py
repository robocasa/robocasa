from robocasa.environments.kitchen.kitchen import *

MEDIUM_HEAT_MIN = np.pi / 4  # ~45 degrees
MEDIUM_HEAT_MAX = 3 * np.pi / 4  # ~135 degrees


class AdjustHeat(Kitchen):
    """
    Adjust Heat: composite task for Sauteing Vegetables activity.

    Simulates the task of adjusting the stove's heat while cooking.

    Steps:
        1) Place the vegetables in the pan.
        2) Identify the correct stove burner then turn the burner knob to adjust the heat from "off" to "medium".
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.5, 0.4))
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
            f"Put the {veg1_lang} and {veg2_lang} in the pan. "
            f"Then turn on the {self.knob.replace('_', ' ')} burner and set the heat to medium."
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
                obj_groups=("pan"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg1",
                obj_groups="vegetable",
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.30, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg2",
                obj_groups="vegetable",
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.30, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg1_in_pan = OU.check_obj_in_receptacle(self, "veg1", "pan")
        veg2_in_pan = OU.check_obj_in_receptacle(self, "veg2", "pan")

        pan_loc = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pan", threshold=0.15
            )
            == self.knob
        )

        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]

        knob_at_medium = MEDIUM_HEAT_MIN <= np.abs(knob_value) <= MEDIUM_HEAT_MAX

        return pan_loc and knob_at_medium and veg1_in_pan and veg2_in_pan
