from robocasa.environments.kitchen.kitchen import *


class WashLettuce(Kitchen):
    """
    WashLettuce: composite task for Making Salads activity.

    Simulates the task of washing salads.

    Steps:
        Turn on the sink. Wash the lettuce under the sink for 25 timesteps

    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = "Wash the lettuce in the sink by running water over it."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.washed_time = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="lettuce",
                obj_groups="lettuce",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                    try_to_place_in="colander",
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()
        if self.sink.check_obj_under_water(self, "lettuce"):
            self.washed_time += 1

    def _check_success(self):
        return self.washed_time >= 25
