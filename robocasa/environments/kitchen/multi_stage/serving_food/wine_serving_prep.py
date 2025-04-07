from robocasa.environments.kitchen.kitchen import *


class WineServingPrep(Kitchen):
    """
    Wine Serving Prep: composite task for Serving Food activity.

    Simulates the task of serving wine.

    Steps:
        Move the wine and the cup from the cabinet to the dining table.

    Restricted to layouts which have a dining table (long counter area
    with stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the wine and
            cup are picked.
    """

    EXCLUDE_LAYOUTS = [0, 2, 4, 5]

    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        alcohol_name = self.get_obj_lang("alcohol")
        cup_name = self.get_obj_lang("cup")
        decoration_name = self.get_obj_lang("decoration")
        ep_meta["lang"] = (
            "Open the cabinet directly in front. "
            f"Then move the {alcohol_name} and the {cup_name} to the counter with the {decoration_name} on it."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="alcohol",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-0.6, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups=["cup", "mug"],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0.6, -1.0),
                ),
            )
        )

        # adding indicator
        cfgs.append(
            dict(
                name="decoration",
                obj_groups="decoration",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        # adding distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=["vegetable", "fruit", "sweets", "dairy"],
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(-1.0, -1.0),
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
        gripper_alcohol_far = OU.gripper_obj_far(self, obj_name="alcohol")
        gripper_cup_far = OU.gripper_obj_far(self, obj_name="cup")
        condiment1_on_counter = OU.check_obj_fixture_contact(
            self, "alcohol", self.dining_table
        )
        condiment2_on_counter = OU.check_obj_fixture_contact(
            self, "cup", self.dining_table
        )

        return (
            gripper_alcohol_far
            and gripper_cup_far
            and condiment1_on_counter
            and condiment2_on_counter
        )
