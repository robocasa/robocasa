from robocasa.environments.kitchen.kitchen import *


class PreheatPot(Kitchen):
    """
    Pre-Heating Pot for Perfect Sauté: composite task for Sauteing Vegetables activity.

    Simulates pre-heating the pot before adding vegetables.

    Steps:
        1. Twist the stove knob to turn on the correct burner for the pot on the stove.
        2. Wait for at least 500 timesteps for the pot to heat up.
        3. After heating, add vegetables to the pot.

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
        veg_lang = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"Turn on the burner for the pot on the stove. "
            f"Wait thirty seconds for the pot to heat up. "
            f"Then add the {veg_lang} to the pot."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.heat_timer = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.15, 0.15),
                    pos=("ref", -0.3),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success condition:
        1. The correct stove burner must be turned on for the designated pot.
        2. The pot must remain on the stove for at least 500 timesteps.
        3. A vegetable must be placed inside the pot.
        """

        pot_on_stove = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pot", threshold=0.15
            )
            == self.knob
        )

        burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)

        if (
            pot_on_stove
            and burner_on
            and OU.check_obj_in_receptacle(self, "vegetable", "pot") is False
        ):
            self.heat_timer += 1

        pot_heated = self.heat_timer >= 500
        if pot_heated is False:
            return False

        vegetable_in_pot = OU.check_obj_in_receptacle(self, "vegetable", "pot")
        return pot_on_stove and burner_on and pot_heated and vegetable_in_pot
