from robocasa.environments.kitchen.kitchen import *


class CreateChildFriendlyFridge(Kitchen):
    """
    Create Child Friendly Fridge: composite task for Loading Fridge.

    Simulates the task of organizing fridge items with child safety in mind.
    alcohol beverages are placed on the top shelf to keep them out of reach
    of children, while fruits and vegetables are placed on lower shelves for
    easy access.

    Steps:
        Place 3 items from the counter near the fridge into the fridge.
        alcohol beverages go on the top shelf (out of child reach).
        Fruits and vegetables go on any shelf except the top shelf.
        The third item is randomly chosen to be either alcohol or fruit/vegetable.
    """

    # certain side-by-side fridges only have 1 shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.third_item_type = self._ep_meta["refs"]["third_item_type"]
        else:
            self.third_item_type = self.rng.choice(["alcohol", "fruit_vegetable"])

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        # Get language descriptions for each item
        item1_lang = self.get_obj_lang("item1")
        item2_lang = self.get_obj_lang("item2")
        item3_lang = self.get_obj_lang("item3")

        if self.third_item_type == "alcohol":
            alcohol_items = [item2_lang, item3_lang]
            fruit_veg_items = [item1_lang]
        else:
            alcohol_items = [item2_lang]
            fruit_veg_items = [item1_lang, item3_lang]

        if len(alcohol_items) == 1:
            alcohol_text = alcohol_items[0]
        else:
            if alcohol_items[0] == alcohol_items[1]:
                alcohol_text = f"{alcohol_items[0]}s"
            else:
                alcohol_text = f"{alcohol_items[0]} and {alcohol_items[1]}"

        if len(fruit_veg_items) == 1:
            fruit_veg_text = fruit_veg_items[0]
        else:
            if fruit_veg_items[0] == fruit_veg_items[1]:
                fruit_veg_text = f"{fruit_veg_items[0]}s"
            else:
                fruit_veg_text = f"{fruit_veg_items[0]} and {fruit_veg_items[1]}"

        alc_pronoun = "it" if len(alcohol_items) == 1 else "them"

        ep_meta["lang"] = (
            f"Place the {alcohol_text} on the top shelf of the fridge "
            f"to keep {alc_pronoun} out of children's reach, and place the {fruit_veg_text} on "
            f"lower shelves for easy access."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["third_item_type"] = self.third_item_type
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="item1",
                obj_groups=("fruit", "vegetable"),
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                        full_depth_region=True,
                    ),
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item2",
                obj_groups="alcohol",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="item1",
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        if self.third_item_type == "fruit_vegetable":
            obj_groups = ("fruit", "vegetable")
        else:
            obj_groups = "alcohol"

        cfgs.append(
            dict(
                name="item3",
                obj_groups=obj_groups,
                init_robot_here=True,
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="item1",
                    size=(0.60, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        item1_on_lower_shelf = self.fridge.check_rack_contact(
            self, "item1", compartment="fridge"
        ) and not self.fridge.check_rack_contact(
            self, "item1", compartment="fridge", rack_index=-1
        )

        item2_on_top_shelf = self.fridge.check_rack_contact(
            self, "item2", compartment="fridge", rack_index=-1
        )

        if self.third_item_type == "alcohol":
            item3_on_top_shelf = self.fridge.check_rack_contact(
                self, "item3", compartment="fridge", rack_index=-1
            )
        else:
            item3_on_lower_shelf = self.fridge.check_rack_contact(
                self, "item3", compartment="fridge"
            ) and not self.fridge.check_rack_contact(
                self, "item3", compartment="fridge", rack_index=-1
            )

        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["item1", "item2", "item3"]
        )

        if not gripper_far:
            return False

        if self.third_item_type == "alcohol":
            return item1_on_lower_shelf and item2_on_top_shelf and item3_on_top_shelf
        else:
            return item1_on_lower_shelf and item2_on_top_shelf and item3_on_lower_shelf
