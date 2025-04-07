from robocasa.environments.kitchen.kitchen import *


class CondimentCollection(Kitchen):
    """
    Condiment Collection: composite task for Clearing Table activity.

    Simulates the task of collecting condiments from the counter and placing
    them in the cabinet.

    Steps:
        Pick the condiments from the counter and place them in the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            condiments are picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("condiment1")
        obj_name_2 = self.get_obj_lang("condiment2")

        ep_meta[
            "lang"
        ] = f"Pick the {obj_name_1} and {obj_name_2} from the counter and place them in the cabinet."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="condiment1",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(0.60, -1.0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(-0.60, -1.0),
                    offset=(-0.05, 0.0),
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
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
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
        gripper_obj_far_condiment1 = OU.gripper_obj_far(self, obj_name="condiment1")
        gripper_obj_far_condiment2 = OU.gripper_obj_far(self, obj_name="condiment2")
        condiment1_inside_cab = OU.obj_inside_of(self, "condiment1", self.cab)
        condiment2_inside_cab = OU.obj_inside_of(self, "condiment2", self.cab)
        return (
            condiment1_inside_cab
            and condiment2_inside_cab
            and gripper_obj_far_condiment1
            and gripper_obj_far_condiment2
        )
