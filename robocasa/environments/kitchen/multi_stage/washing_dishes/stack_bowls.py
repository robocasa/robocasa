from robocasa.environments.kitchen.kitchen import *


class StackBowlsInSink(Kitchen):
    """
    Stack Bowls: composite task for Washing Dishes activity.

    Simulates the task of stacking bowls in the sink.

    Steps:
        Stack the bowls in the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Stack the bowls in the sink."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.65),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.65),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        receptacle1_in_sink = OU.obj_inside_of(self, "receptacle1", self.sink)
        receptacle2_in_sink = OU.obj_inside_of(self, "receptacle2", self.sink)

        receptacle2_in_receptacle1 = OU.check_obj_in_receptacle(
            self, "receptacle2", "receptacle1"
        )
        receptacle1_in_receptacle2 = OU.check_obj_in_receptacle(
            self, "receptacle1", "receptacle2"
        )

        gripper_receptacle1_far = OU.gripper_obj_far(self, obj_name="receptacle1")
        gripper_receptacle2_far = OU.gripper_obj_far(self, obj_name="receptacle2")

        return (
            receptacle1_in_sink
            and receptacle2_in_sink
            and (receptacle2_in_receptacle1 or receptacle1_in_receptacle2)
            and gripper_receptacle1_far
            and gripper_receptacle2_far
        )
