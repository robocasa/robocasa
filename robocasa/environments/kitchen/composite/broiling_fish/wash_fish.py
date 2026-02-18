from robocasa.environments.kitchen.kitchen import *


class WashFish(Kitchen):
    """
    Wash Fish: composite task for Broiling Fish activity.

    Simulates the process of rinsing fish in the sink and placing it on a baking tray.

    Steps:
        Take the fish from the fridge, rinse it in the sink, and place it on the baking tray.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Take the fish out of the fridge, briefly rinse it in the sink, and place it on the tray next to the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)
        self.fish_rinsed = False
        self.fish_water_timer = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="fish",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.20, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor",
                exclude_obj_groups="fish",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.50, 0.3),
                    pos=(0.5, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray",
                obj_groups="oven_tray",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                    rotation=(0),
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()

        if self.sink.check_obj_under_water(self, "obj"):
            self.fish_water_timer += 1

        if self.fish_water_timer >= 25:
            self.fish_rinsed = True
            self.fish_water_timer = 0

    def _check_success(self):
        """
        Checks if the fish has been rinsed for at least 25 timesteps in running water,
        then placed on the tray.
        """

        fish_on_tray = OU.check_obj_in_receptacle(self, "obj", "tray")
        fish_not_grasped = OU.gripper_obj_far(self, "obj")
        success = self.fish_rinsed and fish_on_tray and fish_not_grasped

        return success
