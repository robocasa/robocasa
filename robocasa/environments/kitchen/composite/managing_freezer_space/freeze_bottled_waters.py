from robocasa.environments.kitchen.kitchen import *


class FreezeBottledWaters(Kitchen):
    """
    Freeze Bottled Waters: composite task for Managing Freezer Space activity.

    Simulates the task of freezing bottled waters by taking them from the counter and placing them in the freezer.

    Steps:
        1. Take 2 bottled waters from the counter
        2. Place them in the freezer to freeze
        3. Close the freezer door
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Take the bottled waters from the counter "
            "and place them in the freezer to freeze. Close the freezer door after placing the bottles."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bottle1",
                obj_groups="bottled_water",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.5, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bottle2",
                obj_groups="bottled_water",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="bottle1",
                    size=(0.5, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        bottle1_in_freezer = self.fridge.check_rack_contact(
            self,
            "bottle1",
            compartment="freezer",
        )
        bottle2_in_freezer = self.fridge.check_rack_contact(
            self,
            "bottle2",
            compartment="freezer",
        )

        freezer_door_closed = self.fridge.is_closed(self, compartment="freezer")

        return bottle1_in_freezer and bottle2_in_freezer and freezer_door_closed
