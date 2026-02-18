from robocasa.environments.kitchen.kitchen import *


class FillKettle(Kitchen):
    """
    Fill Kettle: composite task for Boiling activity.

    Simulates the process of filling up a kettle with water from the sink.

    Steps:
        Take the kettle from the cabinet and fill it with water from the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR, ref=self.sink)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the cabinet, pick the kettle from the cabinet, and place it in the sink."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink)

        return obj_in_sink and gripper_obj_far
