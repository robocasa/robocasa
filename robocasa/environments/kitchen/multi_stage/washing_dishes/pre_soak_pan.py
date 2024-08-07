from robocasa.environments.kitchen.kitchen import *


class PreSoakPan(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the pan and sponge and place them into the sink. Then turn on the sink."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("pan"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.4)
                    ),
                    size=(0.35, 0.55),
                    pos=("ref", -1.0),
                ),
                max_size=(0.35, 0.45, None),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("sponge"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]
        pan_in_sink = OU.obj_inside_of(self, "obj1", self.sink)
        sponge_in_sink = OU.obj_inside_of(self, "obj2", self.sink)
        return (
            water_on
            and pan_in_sink
            and sponge_in_sink
            and OU.gripper_obj_far(self, "obj1")
        )
