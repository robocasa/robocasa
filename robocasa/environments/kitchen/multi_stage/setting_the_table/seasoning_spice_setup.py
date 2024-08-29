from robocasa.environments.kitchen.kitchen import *


class SeasoningSpiceSetup(Kitchen):
    """
    Seasoning Spice Setup: composite task for Setting The Table activity.

    Simulates the task of setting the table with seasoning and spices.

    Steps:
        Move the seasoning and spices from the cabinet directly in front to the
        dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the seasoning
            and spices are picked.
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
        condiment1_name = self.get_obj_lang("condiment1")
        condiment2_name = self.get_obj_lang("condiment2")
        ep_meta[
            "lang"
        ] = f"Move the {condiment1_name} and {condiment2_name} from the cabinet directly in front to the dining counter"
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
                name="condiment1",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining2",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_condiment1_far = OU.gripper_obj_far(self, obj_name="condiment1")
        gripper_condiment2_far = OU.gripper_obj_far(self, obj_name="condiment2")
        condiment1_on_counter = OU.check_obj_fixture_contact(
            self, "condiment1", self.dining_table
        )
        condiment2_on_counter = OU.check_obj_fixture_contact(
            self, "condiment2", self.dining_table
        )
        return (
            gripper_condiment1_far
            and gripper_condiment2_far
            and condiment1_on_counter
            and condiment2_on_counter
        )
