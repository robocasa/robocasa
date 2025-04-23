from robocasa.environments.kitchen.kitchen import *


class VeggieDipPrep(Kitchen):
    """
    Veggie Dip Prep: composite task for Snack Preparation activity.

    Simulates the preparation of a vegetable dip snack.

    Steps:
        Place the two vegetables and a bowl onto the tray for setting up a vegetable
        dip station.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place the two vegetables and bowl onto the tray for setting up a vegetable dip station."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        # Tray in the center of the counter
        cfgs.append(
            dict(
                name="tray",
                obj_groups="tray",
                placement=dict(
                    fixture=self.counter,
                    size=(0.3, 0.5),
                    pos=(0, -1),
                ),
            )
        )

        # Two "dippable" vegetables to the left of tray
        cfgs.append(
            dict(
                name="cucumber",
                obj_groups="cucumber",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.5),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="carrot",
                obj_groups="carrot",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.5),
                    pos=(0, -1.0),
                ),
            )
        )

        # Bowl to the right of tray
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.6),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        gripper_far = (
            OU.gripper_obj_far(self, "bowl")
            and OU.gripper_obj_far(self, "cucumber")
            and OU.gripper_obj_far(self, "carrot")
        )
        vegetables_in_tray = OU.check_obj_in_receptacle(
            self, "cucumber", "tray"
        ) and OU.check_obj_in_receptacle(self, "carrot", "tray")
        bowl_in_tray = OU.check_obj_in_receptacle(self, "bowl", "tray")

        return gripper_far and vegetables_in_tray and bowl_in_tray
