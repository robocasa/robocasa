from robocasa.environments.kitchen.kitchen import *


class ClearClutter(Kitchen):
    """
    Clear Clutter: composite task for Washing Fruits And Vegetables activity.

    Simulates the task of washing fruits and vegetables.

    Steps:
        Pick up the fruits and vegetables and place them in the sink turn on the
        sink to wash them. Then, turn the sink off, put them in the tray.
    """

    def __init__(self, *args, **kwargs):
        # internal state variables to keep track of task progress
        self.food_washed = False
        self.washed_time = 0
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        # sample large enough region to place the food items
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.6))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the fruits and vegetables and place them in the sink. "
            "Turn on the sink to wash them. Then turn the sink off and put them in the tray."
        )
        return ep_meta

    def _reset_internal(self):
        self.food_washed = False
        self.washed_time = 0
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        self.num_food = self.rng.choice([1, 2])
        self.num_unwashable = self.rng.choice([1, 2])

        for i in range(self.num_food):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups=["vegetable", "fruit"],
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.40, 0.40),
                        pos=("ref", -1.0),
                    ),
                )
            )

        for i in range(self.num_unwashable):
            cfgs.append(
                dict(
                    name=f"unwashable_obj_{i}",
                    obj_groups="all",
                    # make the object not washable and make sure there aren't 2 trays
                    exclude_obj_groups=["food", "tray"],
                    washable=False,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.40, 0.40),
                        pos=("ref", -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="tray",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.6)
                    ),
                    size=(0.6, 0.8),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        food_in_sink = all(
            [
                OU.obj_inside_of(self, f"obj_{i}", self.sink)
                for i in range(self.num_food)
            ]
        )
        unwashables_not_in_sink = all(
            [
                not OU.obj_inside_of(self, f"unwashable_obj_{i}", self.sink)
                for i in range(self.num_unwashable)
            ]
        )
        water_on = self.sink.get_handle_state(env=self)["water_on"]
        # make sure the food has been washed for suffient time (10 steps)
        if food_in_sink and unwashables_not_in_sink and water_on:
            self.washed_time += 1
            self.food_washed = self.washed_time > 10
        else:
            self.washed_time = 0

        food_in_tray = all(
            [
                OU.check_obj_in_receptacle(self, f"obj_{i}", "receptacle")
                for i in range(self.num_food)
            ]
        )
        unwashables_not_in_tray = all(
            [
                not OU.check_obj_in_receptacle(
                    self, f"unwashable_obj_{i}", "receptacle"
                )
                for i in range(self.num_unwashable)
            ]
        )

        return (
            self.food_washed
            and food_in_tray
            and unwashables_not_in_tray
            and not water_on
        )
