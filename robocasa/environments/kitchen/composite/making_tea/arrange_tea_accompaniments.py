from robocasa.environments.kitchen.kitchen import *


class ArrangeTeaAccompaniments(Kitchen):
    """
    Arrange Tea Accompaniments: composite task for Making Tea activity.

    Simulates the task of arranging tea accompaniments on a tray.
    Steps:
        1. Open the fridge.
        2. Take out milk and other accompaniments.
        3. Place them on a tray on the counter.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.6, 0.6), full_depth_region=True),
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Take out the milk from the fridge and place it on the tray on the counter. Then take the sugar cube near the tray and place it in the tray."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0.0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups="milk",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray",
                obj_groups="tray",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.4), full_depth_region=True
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar_cube",
                obj_groups="sugar_cube",
                object_scale=1.3,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0, -1.0),
                    reuse_region_from="tray",
                ),
            )
        )
        return cfgs

    def _check_success(self):

        milk_on_tray = OU.check_obj_in_receptacle(self, "milk", "tray")
        sugar_cube_on_tray = OU.check_obj_in_receptacle(self, "sugar_cube", "tray")
        tray_gripper_far = OU.gripper_obj_far(self, "tray")
        return milk_on_tray and sugar_cube_on_tray and tray_gripper_far
