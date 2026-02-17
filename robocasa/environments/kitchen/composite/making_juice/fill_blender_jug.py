from robocasa.environments.kitchen.kitchen import *


class FillBlenderJug(Kitchen):
    """
    FillBlenderJug: composite task for the making juice activity.
    Simulates the task of filling the blender jug with water from the sink.
    Steps:
        1. Pick up the blender jug from the counter.
        2. Turn on the sink faucet.
        3. Fill the blender jug with water from the sink.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta[
            "lang"
        ] = "Turn on the sink faucet and place the jug under the water for a few seconds to fill it up."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.num_filled_timesteps = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"blender_jug",
                obj_groups="blender_jug",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()
        if self.sink.check_obj_under_water(self, "blender_jug"):
            self.num_filled_timesteps += 1

    def _check_success(self):
        return self.num_filled_timesteps >= 25
