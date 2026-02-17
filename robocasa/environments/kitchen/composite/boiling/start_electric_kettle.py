from robocasa.environments.kitchen.kitchen import *


class StartElectricKettle(Kitchen):
    """
    Start Electric Kettle: composite task for Boiling activity.

    Simulates the process of starting an electric kettle, along with closing the lid.

    Steps:
        Close the lid of the electric kettle. Then, press down fully to start the kettle.
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
        ep_meta[
            "lang"
        ] = f"Close the electric kettle lid and turn on the electric kettle."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.electric_kettle.set_lid(self, lid_val=1.0)

    def _check_success(self):
        lid_closed = self.electric_kettle.get_state(self)["lid"] <= 0.01

        if not lid_closed:
            return False

        return self.electric_kettle.get_state(self)["turned_on"]
