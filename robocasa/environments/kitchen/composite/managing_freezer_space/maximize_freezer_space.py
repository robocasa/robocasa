from robocasa.environments.kitchen.kitchen import *


class MaximizeFreezerSpace(Kitchen):
    """
    Maximize Freezer Space: composite task for Managing Freezer Space activity.

    Simulates the task of reorganizing freezer items by moving items between the top two racks
    to maximize space efficiency. The task randomly chooses to either move items from the
    second highest rack to the highest rack, or from the highest rack to the second highest rack.

    Steps:
        1. Take items from one rack
        2. Place them on the other rack
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
        self.init_robot_base_ref = self.fridge

        if "refs" in self._ep_meta:
            self.move_direction = self._ep_meta["refs"]["move_direction"]
        else:
            self.move_direction = "up" if self.rng.random() < 0.5 else "down"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        item1_lang = self.get_obj_lang("item1")
        item2_lang = self.get_obj_lang("item2")
        item3_lang = self.get_obj_lang("item3")
        item4_lang = self.get_obj_lang("item4")

        if item1_lang == item2_lang:
            source_items_text = f"{item1_lang}s"
        else:
            source_items_text = f"{item1_lang} and {item2_lang}"

        if item3_lang == item4_lang:
            target_items_text = f"{item3_lang}s"
        else:
            target_items_text = f"{item3_lang} and {item4_lang}"

        if self.move_direction == "up":
            ep_meta["lang"] = (
                f"Take the {source_items_text} from the second highest rack and move them to the highest rack "
                f"where the {target_items_text} are located, to maximize freezer space. Close the freezer door after rearranging."
            )
        else:
            ep_meta["lang"] = (
                f"Take the {source_items_text} from the highest rack and move them to the second highest rack "
                f"where the {target_items_text} are located, to maximize freezer space. Close the freezer door after rearranging."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["move_direction"] = self.move_direction
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        if self.move_direction == "up":
            source_rack = -2
            target_rack = -1
        else:
            source_rack = -1
            target_rack = -2

        cfgs.append(
            dict(
                name="item1",
                obj_groups=("meat", "vegetable", "fruit"),
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=source_rack,
                    ),
                    size=(0.30, 0.20),
                    pos=(-0.15, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item2",
                obj_groups=("meat", "vegetable", "fruit"),
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=source_rack,
                    ),
                    size=(0.30, 0.20),
                    pos=(0.15, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item3",
                obj_groups=("meat", "vegetable", "fruit"),
                exclude_obj_groups=("shrimp"),
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=target_rack,
                    ),
                    size=(0.30, 0.20),
                    pos=(-0.15, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item4",
                obj_groups=("meat", "vegetable", "fruit"),
                exclude_obj_groups=("shrimp"),
                graspable=True,
                freezable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=target_rack,
                    ),
                    size=(0.30, 0.20),
                    pos=(0.15, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        if self.move_direction == "up":
            target_rack = -1
        else:
            target_rack = -2

        item1_on_target_rack = self.fridge.check_rack_contact(
            self, "item1", compartment="freezer", rack_index=target_rack
        )
        item2_on_target_rack = self.fridge.check_rack_contact(
            self, "item2", compartment="freezer", rack_index=target_rack
        )

        item3_still_on_target_rack = self.fridge.check_rack_contact(
            self, "item3", compartment="freezer", rack_index=target_rack
        )
        item4_still_on_target_rack = self.fridge.check_rack_contact(
            self, "item4", compartment="freezer", rack_index=target_rack
        )

        freezer_closed = self.fridge.is_closed(env=self, compartment="freezer")

        return (
            item1_on_target_rack
            and item2_on_target_rack
            and item3_still_on_target_rack
            and item4_still_on_target_rack
            and freezer_closed
        )
