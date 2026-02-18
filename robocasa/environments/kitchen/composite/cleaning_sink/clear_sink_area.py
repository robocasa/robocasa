from robocasa.environments.kitchen.kitchen import *


class ClearSinkArea(Kitchen):
    """
    Clear Sink Area: composite task for Cleaning Sink activity.
    Simulates the process of moving items away from the sink to allow for sink cleaning.

    Steps:
        1. Move the mugs and bowl far away from the sink
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
        ep_meta["lang"] = "Clear the mugs and bowl far away from the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        """
        Configuration of objects in the task. Food items start in the sink, and dishes start on the counter.
        """
        cfgs = []

        # Food items inside the sink
        cfgs.append(
            dict(
                name=f"food",
                obj_groups=["fruit", "vegetable"],
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(),
                    size=(0.30, 0.30),
                    pos=(1.0, 0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="mug1",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="mug2",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _close_to_sink(self, obj_name, dist=1):
        obj = self.objects[obj_name]
        food = self.objects["food"]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        food_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[food.name]])

        obj_dist = np.linalg.norm(obj_pos - food_pos)
        return obj_dist < dist

    def _check_success(self):
        dishes_close = (
            self._close_to_sink("mug1")
            or self._close_to_sink("mug2")
            or self._close_to_sink("bowl")
        )

        gripper_far_food = (
            OU.gripper_obj_far(self, "mug1")
            and OU.gripper_obj_far(self, "mug2")
            and OU.gripper_obj_far(self, "bowl")
        )
        return not dishes_close and gripper_far_food
