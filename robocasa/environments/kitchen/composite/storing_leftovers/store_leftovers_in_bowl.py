from robocasa.environments.kitchen.kitchen import *


class StoreLeftoversInBowl(Kitchen):
    """
    Store Leftovers in Bowl: composite task for Storing Leftovers activity.

    Simulates the process of storing leftover food items (chicken drumstick and a vegetable)
    from plates on the dining counter into a bowl and placing the bowl in the fridge.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

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

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_lang = self.get_obj_lang("vegetable")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                vegetable_lang=vegetable_lang
            )
        else:
            ep_meta["lang"] = (
                f"Pick the chicken drumstick and {vegetable_lang} from their plates "
                f"and place them in the bowl. Then put the bowl in the fridge."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="chicken_drumstick",
                obj_groups="chicken_drumstick",
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
                name="vegetable",
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
                name="bowl",
                obj_groups="bowl",
                init_robot_here=True,
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr1",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(-0.3, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr2",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0.3, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        chicken_in_bowl = OU.check_obj_in_receptacle(self, "chicken_drumstick", "bowl")
        vegetable_in_bowl = OU.check_obj_in_receptacle(self, "vegetable", "bowl")
        bowl_in_fridge = self.fridge.check_rack_contact(self, "bowl")
        gripper_far = OU.gripper_obj_far(self, "bowl")

        return chicken_in_bowl and vegetable_in_bowl and bowl_in_fridge and gripper_far
