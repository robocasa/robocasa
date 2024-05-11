from robocasa.environments.kitchen.kitchen import *


class ManipulateSinkFaucet(Kitchen):
    def __init__(self, behavior="turn_on", *args, **kwargs):
        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior.replace('_', ' ')} the sink faucet"
        return ep_meta

    def _reset_internal(self):

        super()._reset_internal()

        if self.behavior == "turn_on":
            self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        elif self.behavior == "turn_off":
            self.sink.set_handle_state(mode="on", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        
        # distractors
        for i in range(np.random.randint(1, 4)):
            cfgs.append(dict(
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
            ))
        cfgs.append(dict(
            name="distr_sink",
            obj_groups="all",
            washable=True,
            placement=dict(
                fixture=self.sink,
                size=(0.30, 0.40),
                pos=(None, -1.0),
            ),
        ))

        return cfgs

    def _check_success(self):        
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        if "task_refs" in self._ep_meta:
            self.behavior = self._ep_meta["task_refs"]["behavior"]
            self.init_sink_mode = self._ep_meta["task_refs"]["init_sink_mode"]
        else:
            self.behavior = self.rng.choice(["left", "right"])
            self.init_sink_mode = self.rng.choice(["on", "off"])
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"turn the sink spout to the {self.behavior}"
        ep_meta["task_refs"] = dict(
            behavior=self.behavior,
            init_sink_mode=self.init_sink_mode,
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode=self.init_sink_mode, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        
        # distractors
        for i in range(np.random.randint(1, 4)):
            cfgs.append(dict(
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
            ))
        cfgs.append(dict(
            name="distr_sink",
            obj_groups="all",
            washable=True,
            placement=dict(
                fixture=self.sink,
                size=(0.30, 0.40),
                pos=(None, -1.0),
            ),
        ))

        return cfgs

    def _check_success(self):        
        handle_state = self.sink.get_handle_state(env=self)
        success = (handle_state["spout_ori"] == self.behavior)
        
        return success