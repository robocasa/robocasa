from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type, OBJ_CATEGORIES


class PackIdenticalLunches(Kitchen):
    """
    Pack Identical Lunch: composite task for Packing Lunches activity.

    Simulates the task of packing identical lunches with vegetables and meats.
    There are 2 identical vegetables and 2 identical meats, and 2 tupperwares.
    Each tupperware needs to contain 1 vegetable and 1 meat.

    Steps:
        1. Pick vegetables and meats from the fridge.
        2. Place 1 vegetable and 1 meat in each of the two tupperwares on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        self.fridge.open_door(self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg_lang = self.get_obj_lang("vegetable0")
        meat_lang = self.get_obj_lang("meat0")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                veg_lang=veg_lang, meat_lang=meat_lang
            )
        else:
            ep_meta["lang"] = (
                f"Place one {veg_lang} and one {meat_lang} in each tupperware on the nearby counter, "
                "to pack two identical lunches."
            )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        all_veg_categories = get_cats_by_type(
            types=["vegetable"], obj_registries=self.obj_registries
        )
        all_meat_categories = get_cats_by_type(
            types=["meat"], obj_registries=self.obj_registries
        )

        # Filter for graspable vegetables
        veg_categories = []
        for cat in all_veg_categories:
            if cat in OBJ_CATEGORIES:
                for reg in OBJ_CATEGORIES[cat]:
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if hasattr(cat_meta, "graspable") and cat_meta.graspable:
                        veg_categories.append(cat)
                        break

        # Filter for graspable meats (excluding shrimp)
        meat_categories = []
        for cat in all_meat_categories:
            if cat != "shrimp" and cat in OBJ_CATEGORIES:
                for reg in OBJ_CATEGORIES[cat]:
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if hasattr(cat_meta, "graspable") and cat_meta.graspable:
                        meat_categories.append(cat)
                        break

        selected_veg = self.rng.choice(veg_categories, size=1, replace=True)[0]
        selected_meat = self.rng.choice(meat_categories, size=1, replace=True)[0]

        cfgs.append(
            dict(
                name="vegetable0",
                obj_groups=selected_veg,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(-0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=selected_veg,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat0",
                obj_groups=selected_meat,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(-0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat1",
                obj_groups=selected_meat,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware0",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.5, 0.5),
                    pos=(-0.5, -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="tupperware1",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    reuse_region_from="tupperware0",
                    size=(1.5, 0.5),
                    pos=(0.5, -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg_in_0 = []
        meat_in_0 = []
        veg_in_1 = []
        meat_in_1 = []

        for veg in ["vegetable0", "vegetable1"]:
            if OU.check_obj_in_receptacle(self, veg, "tupperware0"):
                veg_in_0.append(veg)
            if OU.check_obj_in_receptacle(self, veg, "tupperware1"):
                veg_in_1.append(veg)

        for meat in ["meat0", "meat1"]:
            if OU.check_obj_in_receptacle(self, meat, "tupperware0"):
                meat_in_0.append(meat)
            if OU.check_obj_in_receptacle(self, meat, "tupperware1"):
                meat_in_1.append(meat)

        tupper0_valid = len(veg_in_0) == 1 and len(meat_in_0) == 1
        tupper1_valid = len(veg_in_1) == 1 and len(meat_in_1) == 1

        all_objs = veg_in_0 + veg_in_1 + meat_in_0 + meat_in_1
        unique_objs = set(all_objs)
        no_duplicates = len(all_objs) == len(unique_objs)
        gripper_far = all(OU.gripper_obj_far(self, obj) for obj in all_objs)

        return tupper0_valid and tupper1_valid and no_duplicates and gripper_far
