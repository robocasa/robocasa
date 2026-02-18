from robocasa.environments.kitchen.kitchen import *


class RinseBowls(Kitchen):
    """
    Rinse Dishes: composite task for Washing Dishes activity.

    Simulates the task of rinsing dishes under running water.

    Steps:
        Pick up a bowl from the counter and hold it under the running faucet.
        Ensure each bowl is rinsed for at least 25 timesteps.
        Once rinsed, place each bowl on the counter.
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
        ep_meta[
            "lang"
        ] = "Rinse each bowl for a few seconds and place them down in the sink or on the counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.dishes = ["dish0", "dish1"]
        self.water_contact_timers = {dish: 0 for dish in self.dishes}
        self.rinsed_objects = {dish: False for dish in self.dishes}

    def _get_obj_cfgs(self):
        cfgs = [
            dict(
                name="dish0",
                obj_groups="bowl",
                object_scale=0.75,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            ),
            dict(
                name="dish1",
                obj_groups="bowl",
                object_scale=0.75,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            ),
        ]
        return cfgs

    def update_state(self):
        super().update_state()

        if not hasattr(self, "water_contact_timers"):
            self.water_contact_timers = {
                obj_name: 0 for obj_name in self.obj_body_id.keys()
            }

        if not hasattr(self, "rinsed_objects"):
            self.rinsed_objects = {
                obj_name: False for obj_name in self.obj_body_id.keys()
            }

        object_names = self.obj_body_id.keys()

        for obj_name in object_names:
            in_water = self.sink.check_obj_under_water(self, obj_name)

            if in_water:
                self.water_contact_timers[obj_name] += 1

                if self.water_contact_timers[obj_name] >= 25:
                    self.rinsed_objects[obj_name] = True
            else:
                if not self.rinsed_objects[obj_name]:
                    self.water_contact_timers[obj_name] = 0

    def _check_success(self):
        """
        Checks if all bowls have been successfully rinsed for at least 25 timesteps
        while the sink's water is running.

        Returns:
            bool: True if all bowls are rinsed for at least 25 timesteps, False otherwise.
        """

        all_rinsed = True
        object_names = self.obj_body_id.keys()
        for obj_name in object_names:
            if not self.rinsed_objects[obj_name]:
                all_rinsed = False

        gripper_far_dish0 = OU.gripper_obj_far(self, obj_name="dish0")
        gripper_far_dish1 = OU.gripper_obj_far(self, obj_name="dish1")

        return all_rinsed and gripper_far_dish0 and gripper_far_dish1
