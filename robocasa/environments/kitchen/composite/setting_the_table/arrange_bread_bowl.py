from robocasa.environments.kitchen.kitchen import *


class ArrangeBreadBowl(Kitchen):
    """
    Arrange Bread Bowl: composite task for Setting The Table activity.

    Simulates the process of placing a bowl of bread on the dining counter.

    Steps:
        1. Pick up the heated bread from the toaster oven.
        2. Place the bread in the bowl next to the toaster oven.
        3. Move the bowl with the bread items to the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )

        if self.toaster_oven.has_multiple_rack_levels():
            self.rack_level = 1
        else:
            self.rack_level = 0

        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        bread_lang = self.get_obj_lang("toaster_oven_bread")

        ep_meta["lang"] = (
            f"Pick up the heated {bread_lang} from the toaster oven and place it in the bowl. "
            "Then move the bread bowl to the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_oven.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="toaster_oven_bread",
                obj_groups="bread_food",
                exclude_obj_groups=("hotdog_bun", "baguette"),
                graspable=True,
                placement=dict(
                    fixture=self.toaster_oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(0.5, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.5, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl_bread",
                obj_groups=("bread_food"),
                exclude_obj_groups=("hotdog_bun"),
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        heated_bread_direct = OU.check_obj_in_receptacle(
            self, "toaster_oven_bread", "bowl"
        )
        bowl_bread_direct = OU.check_obj_in_receptacle(self, "bowl_bread", "bowl")

        if not (heated_bread_direct or bowl_bread_direct):
            breads_in_bowl = False
        else:
            heated_bread_touching = False
            bowl_bread_touching = False

            if heated_bread_direct and not bowl_bread_direct:
                bowl_bread_touching = self.check_contact(
                    self.objects["bowl_bread"], self.objects["toaster_oven_bread"]
                )
            elif bowl_bread_direct and not heated_bread_direct:
                heated_bread_touching = self.check_contact(
                    self.objects["toaster_oven_bread"], self.objects["bowl_bread"]
                )

            heated_bread_ok = heated_bread_direct or heated_bread_touching
            bowl_bread_ok = bowl_bread_direct or bowl_bread_touching
            breads_in_bowl = heated_bread_ok and bowl_bread_ok

        bowl_on_dining = OU.check_obj_fixture_contact(self, "bowl", self.dining_counter)
        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")

        return breads_in_bowl and bowl_on_dining and gripper_far
