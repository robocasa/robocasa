from robocasa.environments.kitchen.kitchen import *


class BeginSlowCooking(Kitchen):
    """
    Begin Slow Cooking: composite task for Slow Cooking activity.

    Simulates the task of starting the slow cooking process for broth by setting the burner to low heat,
    and covering it with a lid.

    Steps:
        1. Set the stove burner to low heat
        2. Place the lid on the saucepan to begin slow cooking
    """

    def __init__(self, stove_id=FixtureType.STOVE, *args, **kwargs):
        self.stove_id = stove_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=self.stove_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        valid_front_knobs = [
            k
            for (k, v) in self.stove.knob_joints.items()
            if v is not None and k.startswith("front")
        ]
        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            self.knob = self.rng.choice(valid_front_knobs)

        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Set the burner covering the saucepan with broth to low heat, "
            "and place the lid on the saucepan to begin slow cooking."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "saucepan", [0.65, 0.42, 0.14, 0.60])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan_with_lid",
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
                auxiliary_obj_enable=True,
                auxiliary_obj_placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        saucepan_loc = self.stove.check_obj_location_on_stove(self, "saucepan")
        saucepan_on_burner = saucepan_loc == self.knob

        # Check that the burner is set to low heat (not just on)
        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state.get(self.knob, 0.0)
        LOW_HEAT_UPPER_THRESHOLD = self.stove.STOVE_HIGH_MIN - 0.00000001
        LOW_HEAT_LOWER_THRESHOLD = self.stove.STOVE_LOW_MIN
        burner_on_low = (
            LOW_HEAT_LOWER_THRESHOLD <= np.abs(knob_value) <= LOW_HEAT_UPPER_THRESHOLD
        )
        lid_on_saucepan = OU.check_obj_in_receptacle(
            self, "saucepan_auxiliary", "saucepan", th=0.02
        )
        gripper_far = OU.gripper_obj_far(self, obj_name="saucepan_auxiliary")

        return all([saucepan_on_burner, burner_on_low, lid_on_saucepan, gripper_far])
