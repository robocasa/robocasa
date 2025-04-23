from robocasa.environments.kitchen.kitchen import *


class DrainVeggies(Kitchen):
    """
    Drain Veggies: composite task for Washing Fruits And Vegetables activity.

    Simulates the task of draining washed vegetables.

    Steps:
        Dump the vegetables from the pot into the sink. Then turn on the sink and
        wash the vegetables. Then turn off the sink and put the vegetables back in
        the pot.
    """

    def __init__(self, *args, **kwargs):
        # internal state variables for the task
        self.vegetables_washed = False
        self.washed_time = 0
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = self.get_obj_lang("obj")
        ep_meta["lang"] = (
            f"Dump the {food_name} from the pot into the sink. Then turn on the water and wash the {food_name}. "
            f"Then turn off the water and put the {food_name} back in the pot."
        )
        return ep_meta

    def _setup_scene(self):
        # reset task progress variables
        self.vegetables_washed = False
        self.washed_time = 0
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"obj",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.6)
                    ),
                    try_to_place_in="pot",
                    size=(0.6, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        vegetables_in_sink = OU.obj_inside_of(self, f"obj", self.sink)
        water_on = self.sink.get_handle_state(env=self)["water_on"]
        # make sure the vegetables are washed for at least 10 steps
        if vegetables_in_sink and water_on:
            self.washed_time += 1
            self.vegetables_washed = self.washed_time > 10
        else:
            self.washed_time = 0

        vegetables_in_pot = OU.check_obj_in_receptacle(self, f"obj", "obj_container")

        return self.vegetables_washed and vegetables_in_pot and not water_on
