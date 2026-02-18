from robocasa.environments.kitchen.kitchen import *


class AddToSoupPot(Kitchen):
    """
    Add To Soup Pot: composite task for Slow Cooking activity.

    Simulates the task of adding vegetables to a simmering soup pot by removing the lid,
    adding two vegetables, and placing the lid back to continue slow cooking.

    Steps:
        1. Remove the lid from the simmering saucepan
        2. Add two vegetables to the saucepan
        3. Place the lid back to continue slow cooking
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

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
            self.vegetable1 = self._ep_meta["refs"]["vegetable1"]
            self.vegetable2 = self._ep_meta["refs"]["vegetable2"]
        else:
            # Choose a front burner for the soup pot
            valid_front_knobs = [
                k
                for (k, v) in self.stove.knob_joints.items()
                if v is not None and k.startswith("front")
            ]
            self.knob = self.rng.choice(valid_front_knobs)

            # Choose vegetables
            vegetable_options = [
                "tomato",
                "mushroom",
                "carrot",
                "potato",
                "sweet_potato",
                "broccoli",
            ]
            self.vegetable1 = self.rng.choice(vegetable_options)
            remaining_options = [v for v in vegetable_options if v != self.vegetable1]
            self.vegetable2 = self.rng.choice(remaining_options)

        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable1 = self.get_obj_lang("vegetable1")
        vegetable2 = self.get_obj_lang("vegetable2")
        ep_meta["lang"] = (
            f"Remove the lid from the simmering saucepan and add the {vegetable1} and {vegetable2} from the plate to the saucepan. "
            "Then place the lid back to continue slow cooking the soup."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        ep_meta["refs"]["vegetable1"] = self.vegetable1
        ep_meta["refs"]["vegetable2"] = self.vegetable2
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "saucepan", [0.65, 0.42, 0.14, 0.60])
        self.stove.set_knob_state(mode="low", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

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
                    ensure_object_boundary_in_range=False,
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
                auxiliary_obj_enable=True,
            )
        )

        cfgs.append(
            dict(
                name="vegetable_plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.4, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=self.vegetable1,
                placement=dict(
                    object="vegetable_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=self.vegetable2,
                placement=dict(
                    object="vegetable_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        saucepan_loc = self.stove.check_obj_location_on_stove(self, "saucepan")
        saucepan_on_burner = saucepan_loc == self.knob

        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state.get(self.knob, 0.0)
        LOW_HEAT_UPPER_THRESHOLD = self.stove.STOVE_HIGH_MIN - 0.00000001
        LOW_HEAT_LOWER_THRESHOLD = self.stove.STOVE_LOW_MIN
        burner_on_low = (
            LOW_HEAT_LOWER_THRESHOLD <= np.abs(knob_value) <= LOW_HEAT_UPPER_THRESHOLD
        )

        vegetable1_in_pot = OU.check_obj_in_receptacle(self, "vegetable1", "saucepan")
        vegetable2_in_pot = OU.check_obj_in_receptacle(self, "vegetable2", "saucepan")

        lid_on_saucepan = OU.check_obj_in_receptacle(
            self, "saucepan_auxiliary", "saucepan", th=0.02
        )

        gripper_far = OU.gripper_obj_far(self, obj_name="saucepan_auxiliary")

        return all(
            [
                saucepan_on_burner,
                burner_on_low,
                vegetable1_in_pot,
                vegetable2_in_pot,
                lid_on_saucepan,
                gripper_far,
            ]
        )
