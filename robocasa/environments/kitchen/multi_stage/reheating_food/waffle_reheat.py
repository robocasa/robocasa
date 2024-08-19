from robocasa.environments.kitchen.kitchen import *


class WaffleReheat(Kitchen):
    """
    Waffle Reheat: composite task for Reheating Food activity.

    Simulates the task of reheating a waffle.

    Steps:
        Open the microwave. Place the bowl with waffle inside the microwave, then
        close the microwave door and turn it on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.6, 0.6), ref=self.microwave)
        )
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Open the microwave, place the bowl with waffle inside the microwave, "
            "then close the microwave door and turn it on."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="waffle",
                obj_groups="waffle",
                placement=dict(
                    fixture=self.counter,
                    size=(0.3, 0.3),
                    pos=(-1.0, -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        gripper_far = OU.gripper_obj_far(self, "waffle")
        waffle_in_bowl = OU.check_obj_in_receptacle(self, "waffle", "waffle_container")
        bowl_in_microwave = OU.obj_inside_of(self, "waffle_container", self.microwave)
        microwave_on = self.microwave.get_state()["turned_on"]
        return waffle_in_bowl and bowl_in_microwave and microwave_on and gripper_far
