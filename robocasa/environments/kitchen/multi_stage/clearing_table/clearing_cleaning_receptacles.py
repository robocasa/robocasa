from robocasa.environments.kitchen.kitchen import *


class ClearingCleaningReceptacles(Kitchen):
    """
    Clearing Cleaning Receptacles: composite task for Clearing Table activity.

    Simulates the process of clearing receptacles from the dining table and
    cleaning them in the sink.

    Steps:
        Pick the receptacles from the dining table and place them in the sink.
        Then, turn on the water.
    """

    EXCLUDE_LAYOUTS = [0, 2, 4, 5]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        # dining_table is a sufficiently large counter closest to the stools
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_pos = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("receptacle1")
        obj_name_2 = self.get_obj_lang("receptacle2")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name_1} and {obj_name_2} and place them in the sink. Then turn on the water."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="receptacle",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="receptacle",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1.0),
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
        receptacle1_in_sink = OU.obj_inside_of(self, "receptacle1", self.sink)
        receptacle2_in_sink = OU.obj_inside_of(self, "receptacle2", self.sink)

        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]

        return receptacle1_in_sink and receptacle2_in_sink and water_on
