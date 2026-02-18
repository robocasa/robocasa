from robocasa.environments.kitchen.kitchen import *


class TurnOnElectricKettle(Kitchen):
    """
    Class encapsulating the atomic turn on electric kettle task.
    """

    EXCLUDE_STYLES = [21, 37, 49]

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["electric_kettle"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Press down the lever to turn on the electric kettle."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _check_success(self):
        return self.electric_kettle.get_state(self)["turned_on"]


class CloseElectricKettleLid(Kitchen):
    """
    Class encapsulating the atomic close electric kettle lid task.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["electric_kettle"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the lid of the electric kettle."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.electric_kettle.set_lid(self, lid_val=1.0)

    def _check_success(self):
        return self.electric_kettle.get_state(self)["lid"] <= 0.01


class OpenElectricKettleLid(Kitchen):
    """
    Class encapsulating the atomic open electric kettle lid task.
    """

    def __init__(self, *args, **kwargs):
        kwargs["enable_fixtures"] = ["electric_kettle"]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Press the button to open the lid of the electric kettle."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _check_success(self):
        return self.electric_kettle.get_state(self)["lid"] >= 0.95
