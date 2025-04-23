from robocasa.environments.kitchen.kitchen import *


class YogurtDelightPrep(Kitchen):
    """
    Yogurt Delight Prep: composite task for Snack Preparation activity.

    Simulates the preparation of a yogurt delight snack.

    Steps:
        Place the yogurt and fruit onto the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # want space for all the objects
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the yogurt and fruit onto the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="yogurt",
                obj_groups="yogurt",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.3),
                    pos=(0, -1),
                ),
            )
        )

        self.num_fruits = self.rng.choice([1, 2, 3])
        for i in range(self.num_fruits):
            cfgs.append(
                dict(
                    name=f"fruit_{i}",
                    obj_groups="fruit",
                    placement=dict(
                        fixture=self.cab,
                        size=(0.5, 0.2),
                        pos=(0, -1),
                        offset=(0.05 * i, 0),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        items_on_counter = all(
            [
                OU.check_obj_fixture_contact(self, f"fruit_{i}", self.counter)
                for i in range(self.num_fruits)
            ]
        )
        items_on_counter = items_on_counter and OU.check_obj_fixture_contact(
            self, "yogurt", self.counter
        )

        objs_far = all(
            [OU.gripper_obj_far(self, f"fruit_{i}") for i in range(self.num_fruits)]
        )
        objs_far = objs_far and OU.gripper_obj_far(self, "yogurt")

        return items_on_counter and objs_far
