from robocasa.environments.kitchen.kitchen import *


class DateNight(Kitchen):
    """
    Date Night: composite task for Setting The Table activity.

    Simulates the task of setting the table for a date night.

    Steps:
        Pick up the decoration and the alcohol from the cabinet and move them to the
        dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the decoration
            and alcohol are picked.
    """

    EXCLUDE_LAYOUTS = [0, 2, 4, 5]

    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        decoration_name = self.get_obj_lang("decoration")
        alcohol_name = self.get_obj_lang("alcohol")
        ep_meta[
            "lang"
        ] = f"Pick up the {decoration_name} and the {alcohol_name} from the cabinet and move them to the dining counter."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="decoration",
                obj_groups="decoration",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="alcohol",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="decoration")
        alcohol_on_dining_table = OU.check_obj_fixture_contact(
            self, "alcohol", self.dining_table
        )
        decoration_on_dining_table = OU.check_obj_fixture_contact(
            self, "decoration", self.dining_table
        )

        return (
            gripper_obj_far and decoration_on_dining_table and alcohol_on_dining_table
        )
