from robocasa.environments.kitchen.kitchen import *


class ClearSink(Kitchen):
    """
    Clear the Sink: composite task for Washing Dishes activity.

    Simulates the process of moving food items out of the sink and moving dishes to the sink.

    Steps:
        1. Remove food items from the sink.
        2. Move the dishes from the counter to the sink to prepare for washing.
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

        dish0_lang = self.get_obj_lang("dish0")
        dish1_lang = self.get_obj_lang("dish1")

        food0_lang = self.get_obj_lang("food0")
        food1_lang = self.get_obj_lang("food1")

        ep_meta["lang"] = (
            f"Remove the {food0_lang} and {food1_lang} from the sink. "
            f"Then move the {dish0_lang} and {dish1_lang} from the counter to the sink for washing."
        )
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
                    name=f"food{i}",
                    obj_groups=["fruit", "vegetable"],
                    graspable=True,
                    placement=dict(
                        fixture=self.sink,
                        size=(0.50, 0.20),
                        pos=(0.5 - (i / 5), 1.0),
                    ),
                )
            )

        # Dishes on the counter
        cfgs.append(
            dict(
                name="dish0",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dish1",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success criteria for the task:
        1. All food items are removed from the sink.
        2. All dishes are moved from the counter into the sink.
        """

        food_on_counter = all(
            [OU.obj_inside_of(self, f"food{i}", self.sink) == False for i in range(2)]
        )

        dishes_in_sink = all(
            [OU.obj_inside_of(self, f"dish{i}", self.sink) for i in range(2)]
        )

        gripper_far_food = all(
            [OU.gripper_obj_far(self, obj_name=f"food{i}") for i in range(2)]
        )
        gripper_far_dishes = all(
            [OU.gripper_obj_far(self, obj_name=f"dish{i}") for i in range(2)]
        )

        return (
            food_on_counter
            and dishes_in_sink
            and gripper_far_food
            and gripper_far_dishes
        )
