from robocasa.environments.kitchen.kitchen import *


class BoilPot(Kitchen):
    """
    Boil Pot: composite task for Boiling activity.

    Simulates the process of boiling water in a pot.

    Steps:
        Take the pot filled with water and place it on the stove.
        Turn on the stove burner to boil the water.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.counter_sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take the pot filled with water and place it on the stove. "
            "Then turn on the stove burner to begin boiling the water."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "pot", [0.5, 0.6, 1.0, 0.3])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        pot_burner_location = self.stove.get_obj_location_on_stove(self, "pot")

        if pot_burner_location is not None:
            burner_on = self.stove.is_burner_on(
                env=self, burner_loc=pot_burner_location
            )
            return burner_on

        return False
