from robocasa.environments.kitchen.kitchen import *


class SoakSponge(Kitchen):
    """
    Soak Sponge: composite task for Washing Dishes activity.

    Simulates the process of soaking a sponge under running water.

    Steps:
        Pick up the sponge and hold it under the running faucet.
        Ensure the sponge is soaked for at least 25 timesteps.
        Once soaked, place the sponge inside the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Soak the sponge with water for a few seconds and then drop it inside the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sponge_contact_time = 0

    def _get_obj_cfgs(self):
        return [
            dict(
                name="obj",
                obj_groups="sponge",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            ),
        ]

    def update_state(self):
        super().update_state()

        water_on = self.sink.get_handle_state(env=self)["water_on"]

        if water_on and self.sink.check_obj_under_water(self, "obj"):
            self.sponge_contact_time += 1

    def _check_success(self):
        """
        Checks if the sponge has been held under running water for at least 25 timesteps.

        Returns:
            bool: True if the sponge is soaked for 25 timesteps, False otherwise.
        """

        gripper_far_sponge = OU.gripper_obj_far(self, obj_name="obj")
        sponge_in_sink = OU.obj_inside_of(self, "obj", self.sink)

        return self.sponge_contact_time >= 25 and gripper_far_sponge and sponge_in_sink
