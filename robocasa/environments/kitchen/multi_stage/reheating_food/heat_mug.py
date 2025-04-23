from robocasa.environments.kitchen.kitchen import *


class HeatMug(Kitchen):
    """
    Heat Mug: composite task for Reheating Food activity.

    Simulates the task of reheating a mug.

    Steps:
        Open the cabinet, pick the mug, place it inside the microwave, and close
        the microwave.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.microwave)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the mug from the cabinet and place it inside the microwave. Then close the microwave."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)
        self.microwave.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
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
        gripper_obj_far = OU.gripper_obj_far(self)
        obj_in_microwave = OU.obj_inside_of(self, "obj", self.microwave)
        door_closed = self.microwave.is_closed(env=self)

        return obj_in_microwave and gripper_obj_far and door_closed
