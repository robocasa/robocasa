from robocasa.environments.kitchen.kitchen import *


class StrainerSetup(Kitchen):
    """Strainer Setup: composite task for Making Tea activity.
    Simulates the task of setting up a tea strainer.
    Steps:
        1. Open the cabinet.
        2. Place the strainer in a bowl upright.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Open the cabinet and setup a tea strainer by placing it in the bowl upright."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="strainer",
                obj_groups="strainer",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        strainer_in_bowl = OU.check_obj_in_receptacle(self, "strainer", "bowl")
        strainer_upright = OU.check_obj_upright(self, "strainer", th=30)
        obj_gripper_far = OU.gripper_obj_far(self, "strainer")
        return strainer_in_bowl and obj_gripper_far and strainer_upright
