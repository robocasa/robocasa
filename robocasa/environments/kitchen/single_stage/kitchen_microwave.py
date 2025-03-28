from robocasa.environments.kitchen.kitchen import *


class MicrowavePressButton(Kitchen):
    """
    Class encapsulating the atomic microwave press button tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            microwave manipulation behavior for the task
    """

    def __init__(self, behavior="turn_on", *args, **kwargs):
        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the microwave tasks
        """
        super()._setup_kitchen_references()
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        if self.behavior == "turn_off":
            self.microwave._turned_on = True
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        if self.behavior == "turn_on":
            ep_meta["lang"] = "press the start button on the microwave"
        elif self.behavior == "turn_off":
            ep_meta["lang"] = "press the stop button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave tasks. This includes the object placement configurations.
        Place the object inside the microwave and on top of another container object inside the microwave

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="all",
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
        """
        Check if the microwave manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
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
