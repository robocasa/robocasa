from robocasa.environments.kitchen.kitchen import *


class PlaceDishesBySink(Kitchen):
    """
    Place Dishes by the Sink: a composite task for Washing Dishes activity.

    Navigate to the counter containing dishes and use the pick-and-place skill
    to transport dishes from the counter to the counter next to the sink.

    Steps:
        1. Locate the cabinet containing the cleaning items.
        2. Open the cabinet door.
        3. Retrieve the first cleaning item and place it near the sink.
        4. Retrieve the second cleaning item and place it near the sink.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        dish0_lang = self.get_obj_lang("dish0")
        dish1_lang = self.get_obj_lang("dish1")
        ep_meta[
            "lang"
        ] = f"Pick the {dish0_lang} and {dish1_lang} from the counter and place them on the counter next to the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="dish0",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dish1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.60, 0.35),
                    pos=("ref", 0.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Success criteria:
        1. Both dishes must be within dist_threshold of the sink (by bbox min‑distance).
        2. Both dishes must be in contact with the counter.
        3. The gripper must be far from both dishes.
        """
        dist_threshold = 0.15

        dish0_dist = OU.obj_fixture_bbox_min_dist(self, "dish0", self.sink)
        dish1_dist = OU.obj_fixture_bbox_min_dist(self, "dish1", self.sink)
        dishes_close = (dish0_dist < dist_threshold) and (dish1_dist < dist_threshold)

        on_counter0 = OU.check_obj_fixture_contact(self, "dish0", self.counter_sink)
        on_counter1 = OU.check_obj_fixture_contact(self, "dish1", self.counter_sink)
        dishes_on_counter = on_counter0 and on_counter1

        gripper_far0 = OU.gripper_obj_far(self, obj_name="dish0")
        gripper_far1 = OU.gripper_obj_far(self, obj_name="dish1")
        gripper_far = gripper_far0 and gripper_far1

        return dishes_close and dishes_on_counter and gripper_far
