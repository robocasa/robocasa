from robocasa.environments.kitchen.kitchen import *


class PushUtensilsToSink(Kitchen):
    """
    Push Utensils To Sink: composite task for Sanitize Surface activity.

    Simulates the task of pushing (since utensils are difficult to grasp)
    utensils into the sink.

    Steps:
        Push the utensils into the sink.
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

        obj1_name = self.get_obj_lang("utensil1")
        obj2_name = self.get_obj_lang("utensil2")

        ep_meta["lang"] = f"Push the {obj1_name} and {obj2_name} into the sink."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="utensil1",
                obj_groups=["utensil"],
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                    offset=(0.07, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups=["utensil"],
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(1.0, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

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
        utensil1_in_sink = OU.obj_inside_of(self, "utensil1", self.sink)
        utensil2_in_sink = OU.obj_inside_of(self, "utensil2", self.sink)
        gripper_utensil1_far = OU.gripper_obj_far(self, obj_name="utensil1")
        gripper_utensil2_far = OU.gripper_obj_far(self, obj_name="utensil2")
        return (
            utensil1_in_sink
            and utensil2_in_sink
            and gripper_utensil1_far
            and gripper_utensil2_far
        )
