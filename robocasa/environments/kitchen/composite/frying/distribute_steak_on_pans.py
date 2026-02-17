from robocasa.environments.kitchen.kitchen import *


class DistributeSteakOnPans(Kitchen):
    """
    Distribute Steak On Pans: composite task for Frying activity.

    Simulates the task of redistributing steaks across pans for better heat distribution
    and maneuverability.

    Steps:
        1) Move 1 steak from the full pan to the empty pan
        2) Turn on the stove burner for the empty pan
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove

        valid_knobs = []
        for knob, joint in self.stove.knob_joints.items():
            if joint is not None and not knob.startswith("rear"):
                valid_knobs.append(knob)

        if "refs" in self._ep_meta:
            self.full_pan_knob = self._ep_meta["refs"]["full_pan_knob"]
            self.empty_pan_knob = self._ep_meta["refs"]["empty_pan_knob"]
        else:
            full_pan_knob_idx = self.rng.choice([0, 1])
            if full_pan_knob_idx == 0:
                self.full_pan_knob = valid_knobs[0]
                self.empty_pan_knob = valid_knobs[1]
            else:
                self.full_pan_knob = valid_knobs[1]
                self.empty_pan_knob = valid_knobs[0]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Turn on the stove burner for the empty pan. "
            "Then move one steak from the full pan to the empty pan "
            "to achieve better heat distribution and more room for maneuverability."
        )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["full_pan_knob"] = self.full_pan_knob
        ep_meta["refs"]["empty_pan_knob"] = self.empty_pan_knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(
            mode="on", knob=self.full_pan_knob, env=self, rng=self.rng
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="full_pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.full_pan_knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="empty_pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.empty_pan_knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="steak1",
                obj_groups="steak",
                placement=dict(
                    object="full_pan",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="steak2",
                obj_groups="steak",
                placement=dict(
                    object="full_pan",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        # Check that both pans are properly placed on stove burners
        full_pan_loc = self.stove.check_obj_location_on_stove(self, "full_pan")
        empty_pan_loc = self.stove.check_obj_location_on_stove(self, "empty_pan")

        pans_on_different_burners = (
            full_pan_loc is not None
            and empty_pan_loc is not None
            and full_pan_loc != empty_pan_loc
        )

        # Check that both burners are turned on
        full_pan_burner_on = self.stove.is_burner_on(
            env=self, burner_loc=self.full_pan_knob
        )
        empty_pan_burner_on = self.stove.is_burner_on(
            env=self, burner_loc=self.empty_pan_knob
        )

        steak1_on_full_pan = OU.check_obj_in_receptacle(self, "steak1", "full_pan")
        steak1_on_empty_pan = OU.check_obj_in_receptacle(self, "steak1", "empty_pan")
        steak2_on_full_pan = OU.check_obj_in_receptacle(self, "steak2", "full_pan")
        steak2_on_empty_pan = OU.check_obj_in_receptacle(self, "steak2", "empty_pan")

        steaks_distributed = (steak1_on_full_pan and steak2_on_empty_pan) or (
            steak1_on_empty_pan and steak2_on_full_pan
        )

        gripper_far = all(
            [
                OU.gripper_obj_far(self, obj_name="steak1"),
                OU.gripper_obj_far(self, obj_name="steak2"),
            ]
        )

        return all(
            [
                pans_on_different_burners,
                full_pan_burner_on,
                empty_pan_burner_on,
                steaks_distributed,
                gripper_far,
            ]
        )
