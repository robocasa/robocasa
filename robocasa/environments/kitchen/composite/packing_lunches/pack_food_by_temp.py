from robocasa.environments.kitchen.kitchen import *


class PackFoodByTemp(Kitchen):
    """
    Pack Food By Temperature: composite task for Packing Lunches activity.

    Simulates the task of packing hot and cold foods into separate tupperwares on the dining counter.

    Steps:
        1. Pick 2 hot items from the stove.
        2. Pick 2 cold items (fruit or vegetable) from the fridge.
        3. Place both hot items in one tupperware and both cold items in another tupperware, both on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.fridge

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id

    def _setup_scene(self):
        self.fridge.open_door(self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        hot0_lang = self.get_obj_lang("hot0")
        hot1_lang = self.get_obj_lang("hot1")
        cold0_lang = self.get_obj_lang("cold0")
        cold1_lang = self.get_obj_lang("cold1")
        ep_meta["lang"] = (
            f"Place the cold items ({cold0_lang} and {cold1_lang}) "
            "from the fridge in one tupperware on the dining counter. "
            f"Place the warm items ({hot0_lang} and {hot1_lang}) "
            "in the stove area in another tupperware. "
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _get_obj_cfgs(self):
        from robocasa.models.objects.kitchen_objects import (
            get_cats_by_type,
            OBJ_CATEGORIES,
        )

        cfgs = []

        # Preselect hot and cold categories
        hot_categories = get_cats_by_type(
            types=["cooked_food"], obj_registries=self.obj_registries
        )
        cold_categories = get_cats_by_type(
            types=["fruit", "vegetable"], obj_registries=self.obj_registries
        )

        # Filter for graspable categories
        graspable_hot_categories = []
        for cat in hot_categories:
            for reg in self.obj_registries:
                if reg in OBJ_CATEGORIES[cat] and OBJ_CATEGORIES[cat][reg].graspable:
                    graspable_hot_categories.append(cat)
                    break

        graspable_cold_categories = []
        for cat in cold_categories:
            for reg in self.obj_registries:
                if reg in OBJ_CATEGORIES[cat] and OBJ_CATEGORIES[cat][reg].graspable:
                    graspable_cold_categories.append(cat)
                    break

        selected_hot = self.rng.choice(graspable_hot_categories, size=2, replace=False)
        selected_cold = self.rng.choice(
            graspable_cold_categories, size=2, replace=False
        )

        cfgs.append(
            dict(
                name="hot0",
                obj_groups=selected_hot[0],
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                    try_to_place_in="pan",
                ),
            )
        )
        cfgs.append(
            dict(
                name="hot1",
                obj_groups=selected_hot[1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="cold0",
                obj_groups=selected_cold[0],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(rack_index=-1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="cold1",
                obj_groups=selected_cold[1],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(rack_index=-2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware0",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.25, 0.45),
                    pos=(0.5, "ref"),
                    rotation=(0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="tupperware1",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.25, 0.45),
                    pos=(-0.5, "ref"),
                    rotation=(0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        hot0_in_0 = OU.check_obj_in_receptacle(self, "hot0", "tupperware0")
        hot1_in_0 = OU.check_obj_in_receptacle(self, "hot1", "tupperware0")
        cold0_in_1 = OU.check_obj_in_receptacle(self, "cold0", "tupperware1")
        cold1_in_1 = OU.check_obj_in_receptacle(self, "cold1", "tupperware1")

        hot0_in_1 = OU.check_obj_in_receptacle(self, "hot0", "tupperware1")
        hot1_in_1 = OU.check_obj_in_receptacle(self, "hot1", "tupperware1")
        cold0_in_0 = OU.check_obj_in_receptacle(self, "cold0", "tupperware0")
        cold1_in_0 = OU.check_obj_in_receptacle(self, "cold1", "tupperware0")

        return (
            (hot0_in_0 and hot1_in_0 and cold0_in_1 and cold1_in_1)
            or (hot0_in_1 and hot1_in_1 and cold0_in_0 and cold1_in_0)
            and OU.gripper_obj_far(self, "hot0")
            and OU.gripper_obj_far(self, "hot1")
            and OU.gripper_obj_far(self, "cold0")
            and OU.gripper_obj_far(self, "cold1")
        )
