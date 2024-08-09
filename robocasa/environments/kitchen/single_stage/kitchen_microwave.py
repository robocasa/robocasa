from robocasa.environments.kitchen.kitchen import *


class MicrowavePressButton(Kitchen):
    def __init__(self, behavior="turn_on", *args, **kwargs):
        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        if self.behavior == "turn_off":
            self.microwave._turned_on = True
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.behavior == "turn_on":
            ep_meta["lang"] = "press the start button on the microwave"
        elif self.behavior == "turn_off":
            ep_meta["lang"] = "press the stop button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="all",
                heatable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        turned_on = self.microwave.get_state()["turned_on"]
        gripper_button_far = self.microwave.gripper_button_far(
            self, button="start_button" if self.behavior == "turn_on" else "stop_button"
        )

        if self.behavior == "turn_on":
            return turned_on and gripper_button_far
        elif self.behavior == "turn_off":
            return not turned_on and gripper_button_far


class TurnOnMicrowave(MicrowavePressButton):
    def __init__(self, behavior=None, *args, **kwargs):
        super().__init__(behavior="turn_on", *args, **kwargs)


class TurnOffMicrowave(MicrowavePressButton):
    def __init__(self, behavior=None, *args, **kwargs):
        super().__init__(behavior="turn_off", *args, **kwargs)
