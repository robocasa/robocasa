from robocasa.environments.kitchen.kitchen import *


class TurnOnToaster(Kitchen):
    """
    Atomic task for pushing toaster lever down to turn on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the toaster tasks
        """
        super()._setup_kitchen_references()
        self.toaster = self.get_fixture(FixtureType.TOASTER)
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        """
        Get the episode metadata for the toaster tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Push down the lever of the toaster to turn it on."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the toaster tasks. This includes the object placement configurations.
        Place the object inside the toaster

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                object_scale=0.85,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the toaster manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        toast_slot = 0
        for slot_pair in range(len(self.toaster.get_state(self).keys())):
            if self.toaster.check_slot_contact(self, "obj", slot_pair):
                toast_slot = slot_pair
                break

        return self.toaster.get_state(self, slot_pair=toast_slot)["turned_on"]
