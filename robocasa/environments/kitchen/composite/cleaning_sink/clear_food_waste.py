from robocasa.environments.kitchen.kitchen import *


class ClearFoodWaste(Kitchen):
    """
    Clear Food Waste: composite task for Cleaning Sink activity.
    Simulates the process of moving food items out of the sink and into a receptacle.

    Steps:
        1. Move the food items out of the sink and into the colander.
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
        ep_meta["lang"] = "Move the food items out of the sink and into the colander."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")

    def _get_obj_cfgs(self):
        cfgs = []

        # Food items inside the sink
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"food_{i}",
                    obj_groups=["fruit", "vegetable"],
                    graspable=True,
                    placement=dict(
                        fixture=self.sink,
                        sample_region_kwargs=dict(),
                        size=(0.30, 0.30),
                        pos=(0.0, 0.5 + 0.2 * i),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="colander",
                obj_groups="colander",
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

        return cfgs

    def _check_success(self):
        food_in_colander = all(
            [
                OU.check_obj_in_receptacle(self, f"food_{i}", "colander")
                for i in range(2)
            ]
        )

        gripper_far_food = all(
            [OU.gripper_obj_far(self, obj_name=f"food_{i}") for i in range(2)]
        )
        return food_in_colander and gripper_far_food
