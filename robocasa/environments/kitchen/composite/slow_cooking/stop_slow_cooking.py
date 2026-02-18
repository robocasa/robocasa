from robocasa.environments.kitchen.kitchen import *


class StopSlowCooking(Kitchen):
    """
    Stop Slow Cooking: composite task for Slow Cooking activity.

    Simulates the task of finishing the slow cooking process by turning off the heat
    and removing the lid from a saucepan, making the broth ready to serve.

    Steps:
        1. Turn off the stove burner to stop the cooking
        2. Remove the lid from the saucepan and place it on the counter

    Args:
        stove_id (int): Enum which serves as a unique identifier for different
            stove types. Used to choose the stove where the saucepan is placed.
    """

    def __init__(self, stove_id=FixtureType.STOVE, *args, **kwargs):
        self.stove_id = stove_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=self.stove_id))

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
        ep_meta[
            "lang"
        ] = "Turn off the stove burner and place the saucepan lid on the counter to prepare serving the broth."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="low")
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
            )
        )

        return cfgs

    def _check_success(self):
        stove_off = not self.stove.is_burner_on(env=self, burner_loc=self.knob)
        lid_on_counter = OU.check_obj_any_counter_contact(self, "saucepan_auxiliary")
        gripper_far = OU.gripper_obj_far(self, obj_name="saucepan_auxiliary")

        return stove_off and lid_on_counter and gripper_far
