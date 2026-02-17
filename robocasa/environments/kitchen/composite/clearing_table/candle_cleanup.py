from robocasa.environments.kitchen.kitchen import *


class CandleCleanup(Kitchen):
    """
    Candle Cleanup: composite task for Clearing Table activity.

    Simulates the process of efficiently clearing the dining table decorations.

    Steps:
        Pick the decorations from the dining table and place it in the open cabinet.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        # dining table is a sufficiently large counter where there are chairs nearby
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")
        ep_meta["lang"] = (
            f"Pick the {obj_name_1} and {obj_name_2} from the dining table "
            f"and place them in the open cabinet. "
            f"Close the cabinet after placing the objects."
        )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj1",
                obj_groups="decoration",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.30),
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    pos=(0, -1),
                    offset=(-0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="decoration",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0, -1),
                    offset=(0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1.0, 0.30),
                    pos=(0.0, 0.0),
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
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_inside_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(self, "obj2", self.cab)
        cab_closed = self.cab.is_closed(env=self)
        return cab_closed and obj1_inside_cab and obj2_inside_cab
