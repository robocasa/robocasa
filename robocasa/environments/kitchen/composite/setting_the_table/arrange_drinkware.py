from robocasa.environments.kitchen.kitchen import *


class ArrangeDrinkware(Kitchen):
    """
    Arrange Drinkware: composite task for Setting The Table activity.

    Simulates the task of arranging drinkware for serving beverages.

    Steps:
        1. Pick up the pitcher from the counter next to the sink.
        2. Place the pitcher on the dining counter.
        3. Pick up the cup from the counter next to the sink.
        4. Place the cup on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.sink_counter = self.register_fixture_ref(
            "sink_counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )

        self.init_robot_base_ref = self.sink_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the pitcher and cup on the dining counter for serving."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pitcher",
                obj_groups="pitcher",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.sink_counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups="cup",
                graspable=True,
                placement=dict(
                    fixture=self.sink_counter,
                    reuse_region_from="pitcher",
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        pitcher_on_dining = OU.check_obj_fixture_contact(
            self, "pitcher", self.dining_counter
        )
        cup_on_dining = OU.check_obj_fixture_contact(self, "cup", self.dining_counter)

        object_names = ["pitcher", "cup"]
        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=name) for name in object_names
        )

        return pitcher_on_dining and cup_on_dining and gripper_far
