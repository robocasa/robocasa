from robocasa.environments.kitchen.kitchen import *


class PrepForTenderizing(Kitchen):
    """
    Prep For Tenderizing: composite task for Meat Preparation activity.

    Simulates the task of preparing meat for tenderizing.

    Steps:
        Retrieve a rolling pin from the cabinet and place it next to the meat on
        the cutting board to prepare for tenderizing.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the rolling pin
            is picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.5))
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Retrieve a rolling pin from the cabinet and place it next to the "
            "meat on the cutting board to prepare for tenderizing."
        )
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
                name="meat",
                graspable=True,
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    size=(0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    pos=(0, -0.3),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        cfgs.append(
            dict(
                name="rolling_pin",
                obj_groups="rolling_pin",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.02),
                    pos=(0, 0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        return (
            OU.check_obj_in_receptacle(self, "rolling_pin", "meat_container")
            and OU.gripper_obj_far(self, obj_name="meat_container")
            and OU.check_obj_in_receptacle(self, "meat", "meat_container")
        )
