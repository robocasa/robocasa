from robocasa.environments.kitchen.kitchen import *


class RemoveSteamedVegetables(Kitchen):
    """
    Remove Steamed Vegetables: composite task for Steaming Vegetables activity.

    Simulates the task of removing steamed vegetables and placing them in a plate.

    Steps:
        1. Take the steamed vegetables out of the saucepan
        2. Place the vegetables in a plate on the counter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        # Choose a front burner for the saucepan
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
        veggie1 = self.get_obj_lang("vegetable1")
        veggie2 = self.get_obj_lang("vegetable2")

        if veggie1 == veggie2:
            vegetables_text = f"{veggie1}s"
        else:
            vegetables_text = f"{veggie1} and {veggie2}"

        ep_meta[
            "lang"
        ] = f"The {vegetables_text} are done steaming. Take them out of the saucepan and place them on the plate on the counter."
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
                name="saucepan",
                obj_groups="saucepan",
                object_scale=[1.35, 1.35, 0.75],
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=vegetable_options,
                placement=dict(
                    object="saucepan",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=vegetable_options,
                placement=dict(
                    object="saucepan",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="serving_plate",
                obj_groups="plate",
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

        return cfgs

    def _check_success(self):
        veggie1_in_plate = OU.check_obj_in_receptacle(
            self, "vegetable1", "serving_plate"
        )
        veggie2_in_plate = OU.check_obj_in_receptacle(
            self, "vegetable2", "serving_plate"
        )
        gripper_far_plate = OU.gripper_obj_far(self, "serving_plate")

        return veggie1_in_plate and veggie2_in_plate and gripper_far_plate
