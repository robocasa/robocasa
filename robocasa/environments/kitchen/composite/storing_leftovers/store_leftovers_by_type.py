from robocasa.environments.kitchen.kitchen import *


class StoreLeftoversByType(Kitchen):
    """
    Store Leftovers by Type: composite task for Storing Leftovers activity.

    Simulates the process of storing leftover food items (meat and vegetable)
    from plates on the dining counter into the fridge, ensuring meats and vegetables
    are stored on different racks at the same level as existing items of the same type.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    # these styles only have 1 rack for fridge_side_by_side
    EXCLUDED_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "stool1" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
        else:
            registered_stool_ids = set()
            self.stool1 = None
            self.stool2 = None

            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                        fixture_id = id(fixture)
                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)
                            if self.stool1 is None:
                                self.stool1 = fixture
                            elif self.stool2 is None:
                                self.stool2 = fixture
                                break

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )

        if "meat_rack_index" in self._ep_meta:
            self.meat_rack_index = self._ep_meta["meat_rack_index"]
            self.vegetable_rack_index = self._ep_meta["vegetable_rack_index"]
        else:
            choices = [-1, -2]
            meat_choice = int(self.rng.choice(choices))
            veg_choice = int(-2 if meat_choice == -1 else -1)
            self.meat_rack_index = meat_choice
            self.vegetable_rack_index = veg_choice

        self.init_robot_base_ref = self.stool1

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_lang = self.get_obj_lang("vegetable_counter")
        meat_lang = self.get_obj_lang("meat_counter")

        ep_meta["meat_rack_index"] = self.meat_rack_index
        ep_meta["vegetable_rack_index"] = self.vegetable_rack_index

        ep_meta["lang"] = (
            f"Transport the {meat_lang} and {vegetable_lang} to the fridge. "
            f"Place the {meat_lang} on the rack containing meat "
            f"and place the {vegetable_lang} on the rack containing the vegetable, "
            f"ensuring meats and vegetables are on different racks."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat_counter",
                obj_groups="meat",
                exclude_obj_groups=["shrimp"],
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable_counter",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat_fridge",
                obj_groups="meat",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(-0.3, 1.0),
                    sample_region_kwargs=dict(
                        rack_index=self.meat_rack_index,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable_fridge",
                obj_groups="vegetable",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0.3, 1.0),
                    sample_region_kwargs=dict(
                        rack_index=self.vegetable_rack_index,
                    ),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        # Meat checks on the correct rack
        meat_on_counter_rack = self.fridge.check_rack_contact(
            self, "meat_counter", rack_index=self.meat_rack_index
        )
        meat_on_target_rack = self.fridge.check_rack_contact(
            self, "meat_fridge", rack_index=self.meat_rack_index
        )

        # Vegetable checks on the correct (different) rack
        veg_on_counter_rack = self.fridge.check_rack_contact(
            self, "vegetable_counter", rack_index=self.vegetable_rack_index
        )
        veg_on_target_rack = self.fridge.check_rack_contact(
            self, "vegetable_fridge", rack_index=self.vegetable_rack_index
        )

        # Ensure robot has released both items
        gripper_far_meat = OU.gripper_obj_far(self, "meat_counter")
        gripper_far_vegetable = OU.gripper_obj_far(self, "vegetable_counter")

        return all(
            [
                meat_on_counter_rack,
                meat_on_target_rack,
                veg_on_counter_rack,
                veg_on_target_rack,
                gripper_far_meat,
                gripper_far_vegetable,
            ]
        )
