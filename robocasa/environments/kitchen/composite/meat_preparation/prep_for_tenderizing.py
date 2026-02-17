from robocasa.environments.kitchen.kitchen import *


class PrepForTenderizing(Kitchen):
    """
    Prep For Tenderizing: composite task for Meat Preparation activity.

    Simulates the task of preparing meat for tenderizing.

    Steps:
        Retrieve a rolling pin from the cabinet and place it next to the meat on
        the cutting board to prepare for tenderizing.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.9, 0.6))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the cabinet, retrieve the rolling pin, and place it next to the "
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
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
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
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
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
