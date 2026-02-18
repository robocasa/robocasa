from robocasa.environments.kitchen.kitchen import *


class MoveFridgeToFreezer(Kitchen):
    """
    Move Fridge To Freezer: composite task for Managing Freezer Space.

    Simulates the task of moving refrigerated items from the fridge to the freezer
    for freezing or long-term storage.

    Steps:
        1. Pick up the refrigerated item from the fridge
        2. Place the item in the freezer
        3. Close the fridge and freezer doors
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        item_lang = self.get_obj_lang("refrigerated_item")

        ep_meta["lang"] = (
            f"There's a {item_lang} in the fridge that needs to be moved "
            f"to the freezer for freezing. Pick up the {item_lang} from the fridge, "
            f"place it on any shelf in the freezer, and close the fridge and freezer doors when done."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")
        self.fridge.open_door(env=self, compartment="fridge")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="refrigerated_item",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
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
                    sample_region_kwargs=dict(
                        compartment="fridge",
                    ),
                    size=(0.4, 0.2),
                    pos=(-0.3, -0.5),
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
                    pos=(0.3, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        item_in_freezer = self.fridge.check_rack_contact(
            self,
            "refrigerated_item",
            compartment="freezer",
        )

        freezer_closed = self.fridge.is_closed(self, compartment="freezer")
        fridge_closed = self.fridge.is_closed(self, compartment="fridge")

        fridge_distractor_in_place = self.fridge.check_rack_contact(
            self, "fridge_distractor", compartment="fridge"
        )

        freezer_distractor_in_place = self.fridge.check_rack_contact(
            self,
            "freezer_distractor",
            compartment="freezer",
        )

        return (
            item_in_freezer
            and freezer_closed
            and fridge_closed
            and fridge_distractor_in_place
            and freezer_distractor_in_place
        )
