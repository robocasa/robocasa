from robocasa.environments.kitchen.kitchen import *


class ClearSinkSpace(Kitchen):
    def __init__(self, *args, **kwargs):
        self.num_sink_objs = -1
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink
        if "refs" in self._ep_meta:
            self.num_sink_objs = self._ep_meta["refs"]["num_sink_objs"]
        else:
            self.num_sink_objs = int(self.rng.choice([1, 2]))

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the objects in the sink and place them on the counter to clear sink space for washing produce."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_sink_objs"] = self.num_sink_objs
        return ep_meta

    def _get_obj_cfgs(self):

        cfgs = []
        for i in range(self.num_sink_objs):
            cfgs.append(
                dict(
                    name=f"obj{i}",
                    exclude_obj_groups=["food"],
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.sink,
                        size=(0.40, 0.30),
                        pos=(0.0, 1.0),
                    ),
                )
            )
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"food_item{i}",
                    obj_groups=("fruit", "vegetable"),
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.35, 0.35),
                        pos=("ref", 0.0),
                        sample_region_kwargs=dict(ref=self.counter, loc="left_right"),
                    ),
                )
            )
        return cfgs

    def _check_success(self):

        gripper_objs_far = all(
            [OU.gripper_obj_far(self, f"obj{i}") for i in range(self.num_sink_objs)]
        )
        objs_on_counter = all(
            [
                OU.check_obj_fixture_contact(self, f"obj{i}", self.counter)
                for i in range(self.num_sink_objs)
            ]
        )
        return objs_on_counter and gripper_objs_far
