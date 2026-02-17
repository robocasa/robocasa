from robocasa.environments.kitchen.kitchen import *


class SteamVeggiesWithWater(Kitchen):
    """
    Steam Veggies With Water: composite task for Steaming Vegetables activity.

    Simulates the task of placing vegetables in a saucepan with water and putting the lid on for steaming.

    Steps:
        1. Pick up two different vegetables from the plate
        2. Place the vegetables in the saucepan with water
        3. Put the saucepan lid on
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
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

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veggie1 = self.get_obj_lang("vegetable1")
        veggie2 = self.get_obj_lang("vegetable2")

        if veggie1 == veggie2:
            vegetables_text = f"{veggie1}s"
        else:
            vegetables_text = f"{veggie1} and {veggie2}"

        ep_meta[
            "lang"
        ] = f"Place the {vegetables_text} in the saucepan with water and put the lid on to begin steaming the vegetables."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "saucepan", [0.5, 0.6, 1.0, 0.3])
        self.stove.set_knob_state(knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        vegetable_options = [
            "broccoli",
            "carrot",
            "potato",
            "sweet_potato",
        ]

        cfgs.append(
            dict(
                name="vegetable_plate",
                obj_groups="plate",
                init_robot_here=True,
                placement=dict(
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

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=vegetable_options,
                placement=dict(
                    object="vegetable_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=vegetable_options,
                placement=dict(
                    object="vegetable_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan_with_lid",
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    ensure_object_boundary_in_range=False,
                ),
                auxiliary_obj_enable=True,
                auxiliary_obj_placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.6, 0.55),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veggie1_in_saucepan = OU.check_obj_in_receptacle(self, "vegetable1", "saucepan")
        veggie2_in_saucepan = OU.check_obj_in_receptacle(self, "vegetable2", "saucepan")

        lid_on_saucepan = OU.check_obj_in_receptacle(
            self, "saucepan_auxiliary", "saucepan", th=0.02
        )

        saucepan_loc = self.stove.check_obj_location_on_stove(self, "saucepan")
        saucepan_on_burner = saucepan_loc == self.knob

        gripper_far_lid = OU.gripper_obj_far(self, "saucepan_auxiliary")

        return (
            veggie1_in_saucepan
            and veggie2_in_saucepan
            and lid_on_saucepan
            and saucepan_on_burner
            and gripper_far_lid
        )
