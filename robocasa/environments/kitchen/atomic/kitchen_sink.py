from robocasa.environments.kitchen.kitchen import *


class ManipulateSinkFaucet(Kitchen):
    """
    Class encapsulating the atomic manipulate sink faucet tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            sink faucet manipulation behavior for the task.
    """

    def __init__(self, behavior="turn_on", *args, **kwargs):

        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the sink faucet tasks
        """
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink faucet tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior.replace('_', ' ').capitalize()} the sink faucet."
        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the sink faucet tasks.
        This includes setting the sink faucet state based on the behavior
        """
        super()._setup_scene()

        if self.behavior == "turn_on":
            self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        elif self.behavior == "turn_off":
            self.sink.set_handle_state(mode="on", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink faucet tasks. This includes the object placement configurations.
        Place the objects on the counter and sink as distractors.

        Returns:
            list: List of object configurations
        """
        cfgs = []

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the sink faucet manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]

        if self.behavior == "turn_on":
            success = water_on
        elif self.behavior == "turn_off":
            success = not water_on

        return success


class TurnOnSinkFaucet(ManipulateSinkFaucet):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="turn_on", *args, **kwargs)


class TurnOffSinkFaucet(ManipulateSinkFaucet):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="turn_off", *args, **kwargs)


class TurnSinkSpout(Kitchen):
    """
    Class encapsulating the atomic turn sink spout tasks.

    Args:
        behavior (str): "left" or "right". Used to define the desired sink spout
        manipulation behavior for the task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the sink spout tasks
        """
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        if "task_refs" in self._ep_meta:
            self.behavior = self._ep_meta["task_refs"]["behavior"]
            self.init_sink_mode = self._ep_meta["task_refs"]["init_sink_mode"]
        else:
            self.behavior = self.rng.choice(["left", "right"])
            self.init_sink_mode = self.rng.choice(["on", "off"])
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink spout tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Turn the sink spout to the {self.behavior}."
        ep_meta["task_refs"] = dict(
            behavior=self.behavior,
            init_sink_mode=self.init_sink_mode,
        )
        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the sink spout tasks.
        This includes setting the sink spout state based on the behavior
        """
        super()._setup_scene()
        self.sink.set_handle_state(mode=self.init_sink_mode, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink spout tasks. This includes the object placement configurations.
        Place the objects on the counter and sink as distractors.
        """
        cfgs = []

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the sink spout manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        handle_state = self.sink.get_handle_state(env=self)
        success = handle_state["spout_ori"] == self.behavior

        return success


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

        if "initial_temp_state" not in self._ep_meta:
            self.initial_temp_state = self.rng.choice(["cold", "hot"])
        else:
            self.initial_temp_state = self._ep_meta["initial_temp_state"]
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
