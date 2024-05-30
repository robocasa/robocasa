from robocasa.environments.kitchen.kitchen import *


class ThawInSink(Kitchen):
    def __init__(self, obj_groups="all", *args, **kwargs):
        self.obj_groups = obj_groups
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.4, 0.4)))
        self.init_robot_base_pos = self.sink

    def _reset_internal(self):

        super()._reset_internal()

        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        ep_meta["lang"] = f"Pick the {obj_name} from the counter and place it in the sink. Then turn on the sink faucet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            graspable=True,
            washable=True,
            freezable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.4, 0.4)

                ),
                try_to_place_in="container",
                size=(0.30, 0.40),
                pos=("ref", -1.0),
            ),
        ))
        
        # distractors
        cfgs.append(dict(
            name="distr_counter",
            obj_groups="all",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                offset=(0.0, 0.30),
            ),
        ))


        return cfgs

    def _check_success(self):
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink)
        
        handle_state = self.sink.get_handle_state(env=self)        
        water_on = handle_state["water_on"]

        return obj_in_sink and water_on


        