from robocasa.environments.kitchen.kitchen import *


class TurnOnToaster(Kitchen):
    """
    Class encapsulating the atomic microwave press button tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            microwave manipulation behavior for the task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the microwave tasks
        """
        super()._setup_kitchen_references()
        self.toaster = self.get_fixture(FixtureType.TOASTER)
        self.init_robot_base_pos = self.toaster

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "turn on the toaster"
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
                obj_groups=(
                    "/home/soroushn/code/robocasa-dev/robocasa/models/assets/objects/objaverse/bread/bread_3/model.xml"
                ),
                placement=dict(
                    fixture=self.toaster,
                    ensure_valid_placement=True,
                    rotation=(-np.pi / 2, -np.pi / 2),
                    rotation_axis="y",
                ),
                object_scale=1.0,
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the microwave manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        turned_on = self.toaster._turned_on
        return turned_on
