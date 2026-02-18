from robocasa.environments.kitchen.kitchen import *


class MoveFreezerToFridge(Kitchen):
    """
    Move Freezer To Fridge: composite task for Loading Fridge.

    Simulates the task of moving frozen meat from the freezer to the fridge
    to start the defrosting process.

    Steps:
        1. Pick up the frozen meat from the freezer
        2. Place the frozen meat in the fridge
        3. Close the fridge and freezer doors
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")
        self.fridge.open_door(env=self, compartment="fridge")

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("frozen_meat")

        ep_meta["lang"] = (
            f"Pick up the frozen {meat_lang} from the freezer, "
            f"place it on any shelf in the fridge, and close the fridge and freezer doors when done."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="frozen_meat",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="freezer_distractor",
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                    ),
                    size=(0.4, 0.2),
                    pos=(-0.3, -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0.3, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_in_fridge = self.fridge.check_rack_contact(
            self,
            "frozen_meat",
            compartment="fridge",
        )

        freezer_closed = self.fridge.is_closed(self, compartment="freezer")
        fridge_closed = self.fridge.is_closed(self, compartment="fridge")

        freezer_distractor_in_place = self.fridge.check_rack_contact(
            self, "freezer_distractor", compartment="freezer"
        )

        fridge_distractor_in_place = self.fridge.check_rack_contact(
            self,
            "fridge_distractor",
            compartment="fridge",
        )

        return (
            meat_in_fridge
            and freezer_closed
            and fridge_closed
            and freezer_distractor_in_place
            and fridge_distractor_in_place
        )
