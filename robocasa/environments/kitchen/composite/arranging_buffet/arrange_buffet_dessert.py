from robocasa.environments.kitchen.kitchen import *


class ArrangeBuffetDessert(Kitchen):
    """
    Arrange Buffet Dessert: composite task for Arranging Buffet.

    Simulates taking sweets from the fridge and placing them on plates on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        sweet1 = self.get_obj_lang("sweet1")
        sweet2 = self.get_obj_lang("sweet2")

        if sweet1 == sweet2:
            ep_meta["lang"] = (
                f"Take the {sweet1}s from the fridge and place them on the empty tray "
                f"on the dining counter."
            )
        else:
            ep_meta["lang"] = (
                f"Take the {sweet1} and {sweet2} from the fridge and place them on the empty tray "
                f"on the dining counter."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="sweet1",
                obj_groups="sweets",
                exclude_obj_groups=("marshmallow", "sugar_cube", "cookie_dough_ball"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                    size=(0.30, 0.20),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sweet2",
                obj_groups="sweets",
                exclude_obj_groups=("marshmallow", "sugar_cube", "cookie_dough_ball"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                    size=(0.30, 0.20),
                    pos=(0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray",
                obj_groups="tray",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.40, 0.40),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_cooked_food",
                obj_groups="cooked_food",
                exclude_obj_groups=("pizza"),
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                    try_to_place_in="tray",
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_bread_food",
                obj_groups="bread_food",
                exclude_obj_groups=("sweets"),
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                    try_to_place_in="tray",
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_glass_cup",
                obj_groups="glass_cup",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_glass_cup_2",
                obj_groups="glass_cup",
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="distractor_glass_cup",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sweet1_on_tray = OU.check_obj_in_receptacle(self, "sweet1", "tray")
        sweet2_on_tray = OU.check_obj_in_receptacle(self, "sweet2", "tray")

        tray_on_counter = OU.check_obj_fixture_contact(
            self, "tray", self.dining_counter
        )

        gripper_far = (
            OU.gripper_obj_far(self, obj_name="sweet1", th=0.15)
            and OU.gripper_obj_far(self, obj_name="sweet2", th=0.15)
            and OU.gripper_obj_far(self, obj_name="tray", th=0.15)
        )

        return sweet1_on_tray and sweet2_on_tray and tray_on_counter and gripper_far
