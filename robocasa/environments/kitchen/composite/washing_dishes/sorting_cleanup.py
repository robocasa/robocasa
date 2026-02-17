from robocasa.environments.kitchen.kitchen import *


class SortingCleanup(Kitchen):
    """
    Sorting Cleanup: composite task for Washing Dishes activity.

    Simulates the task of sorting and cleaning dishes.

    Steps:
        Pick the mug and place it in the sink. Pick the bowl and place it in the
        cabinet and then close the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR, ref=self.sink)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )

        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the mug and place it in the sink. "
            "Pick the bowl and place it in the open cabinet and then close the cabinet."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        # not fully open since it may come in contact with eef
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="mug",
                obj_groups=("mug"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.2, 0.4),
                    pos=("ref", -1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="bowl",
                obj_groups=("bowl"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        # large enough region to sample the bowl
                        top_size=(0.5, 0.5),
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        mug_in_sink = OU.obj_inside_of(self, "mug", self.sink)
        bowl_in_cab = OU.obj_inside_of(self, "bowl", self.cab)
        cab_closed = self.cab.is_closed(env=self)
        return (
            mug_in_sink
            and bowl_in_cab
            and cab_closed
            and OU.gripper_obj_far(self, "mug")
        )
