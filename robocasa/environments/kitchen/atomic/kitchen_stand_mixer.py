from robocasa.environments.kitchen.kitchen import *


class OpenStandMixerHead(Kitchen):
    """
    Class encapsulating the atomic stand mixer head tasks.
    """

    def __init__(self, *args, **kwargs):
        kwargs["enable_fixtures"] = ["stand_mixer"]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stand_mixer = self.get_fixture(FixtureType.STAND_MIXER)
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the stand mixer head."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _check_success(self):
        """
        Check if the stand mixer head is open.

        Returns:
            bool: True if the head is open, False otherwise.
        """
        return self.stand_mixer.get_state(self)["head"] > 0.99


class CloseStandMixerHead(Kitchen):
    """
    Class encapsulating the atomic stand mixer head tasks.
    """

    def __init__(self, *args, **kwargs):
        kwargs["enable_fixtures"] = ["stand_mixer"]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stand_mixer = self.get_fixture(FixtureType.STAND_MIXER)
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the stand mixer head."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stand_mixer.set_head_pos(self)

    def _check_success(self):
        """
        Check if the stand mixer head is closed.

        Returns:
            bool: True if the head is closed, False otherwise.
        """
        return self.stand_mixer.get_state(self)["head"] < 0.01
