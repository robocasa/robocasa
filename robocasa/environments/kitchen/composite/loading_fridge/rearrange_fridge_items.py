from robocasa.environments.kitchen.kitchen import *


class RearrangeFridgeItems(Kitchen):
    """
    Rearrange Fridge Items: composite task for Loading Fridge.

    Simulates the task of rearranging items in the fridge to make space.
    There are 2 items on either the top shelf or second highest shelf that need
    to be moved to the other shelf so that a pitcher on the counter can be placed
    on the cleared shelf.

    Steps:
        Move the 2 items from the source shelf to the destination shelf,
        then place the pitcher from the counter on the cleared shelf.
    """

    # certain side-by-side fridges only have 1 shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(
        self, obj_registries=("aigen", "objaverse", "lightwheel"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.source_rack_index = self._ep_meta["refs"]["source_rack_index"]
            self.dest_rack_index = self._ep_meta["refs"]["dest_rack_index"]
        else:
            self.source_rack_index = -1 if self.rng.random() < 0.5 else -2
            self.dest_rack_index = -2 if self.source_rack_index == -1 else -1

        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        source_shelf = (
            "top shelf" if self.source_rack_index == -1 else "second highest shelf"
        )
        dest_shelf = (
            "top shelf" if self.dest_rack_index == -1 else "second highest shelf"
        )

        item1_lang = self.get_obj_lang("item1_source")
        item2_lang = self.get_obj_lang("item2_source")

        ep_meta["lang"] = (
            f"Move the {item1_lang} and {item2_lang} from the {source_shelf} to the {dest_shelf} "
            f"to make space. Then, place the pitcher from the counter on the cleared {source_shelf}."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["source_rack_index"] = self.source_rack_index
        ep_meta["refs"]["dest_rack_index"] = self.dest_rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pitcher",
                obj_groups="pitcher",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                        full_depth_region=True,
                    ),
                    size=(0.60, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item1_source",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.source_rack_index,
                    ),
                    size=(0.4, 0.2),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item2_source",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.source_rack_index,
                    ),
                    size=(0.4, 0.2),
                    pos=(0.3, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        pitcher_on_source_rack = self.fridge.check_rack_contact(
            self, "pitcher", compartment="fridge", rack_index=self.source_rack_index
        )

        items_on_dest_rack = all(
            self.fridge.check_rack_contact(
                self, item_name, compartment="fridge", rack_index=self.dest_rack_index
            )
            for item_name in ["item1_source", "item2_source"]
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["pitcher", "item1_source", "item2_source"]
        )

        return pitcher_on_source_rack and items_on_dest_rack and gripper_far
