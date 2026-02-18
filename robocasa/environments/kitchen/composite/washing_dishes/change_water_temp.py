from robocasa.environments.kitchen.kitchen import *


class AdjustWaterTemperature(Kitchen):
    """
    Change Water Temperature: composite task for Washing Dishes activity.

    Simulates the process of adjusting the water temperature in the sink.

    Steps:
        1) Determine the current water temperature using the sink handle state.
        2) Adjust the sink handle to the opposite temperature state.
        3) Make sure the water is still running.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_ref = self.sink

        self.initial_temp_state = self.rng.choice(["cold", "hot"])
        if self.initial_temp_state == "cold":
            self.target_temp_state = "hot"
        else:
            self.target_temp_state = "cold"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"The water running in the sink is {self.initial_temp_state}. "
            f"Adjust the faucet handle to run the water {self.target_temp_state}. "
            "Make sure to keep the water on."
        )
        ep_meta["initial_temp_state"] = self.initial_temp_state
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(self, self.rng, mode="on")
        if self.initial_temp_state == "cold":
            self.sink.set_temp_state(self, self.rng, min_temp=0.30, max_temp=0.40)
        elif self.initial_temp_state == "hot":
            self.sink.set_temp_state(self, self.rng, min_temp=0.60, max_temp=0.70)

    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        water_running = handle_state["water_on"]
        curr_temp = handle_state["water_temp"]
        if self.target_temp_state == "hot" and curr_temp >= 0.60 and water_running:
            return True
        elif self.target_temp_state == "cold" and curr_temp <= 0.40 and water_running:
            return True
        else:
            return False
