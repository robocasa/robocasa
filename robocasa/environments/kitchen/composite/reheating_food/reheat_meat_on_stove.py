from robocasa.environments.kitchen.kitchen import *


class ReheatMeatOnStove(Kitchen):
    """
    Reheat Meat On Stove: composite task for Reheating Food activity.

    Simulates the task of reheating non-warm meat by placing it on a pan with low heat.

    Steps:
        1. Take the meat from the plate
        2. Place the meat on the pan on the stove
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.dining_counter

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
        meat_lang = self.get_obj_lang("meat")
        ep_meta[
            "lang"
        ] = f"The cooked {meat_lang} has gone cold. Place it on the pan on the stove to reheat it."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="low")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_food",
                obj_groups=("vegetable", "fruit"),
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan")

        pan_on_correct_burner = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pan", threshold=0.15
            )
            == self.knob
        )

        gripper_away = OU.gripper_obj_far(self, "meat")

        return meat_in_pan and pan_on_correct_burner and gripper_away
