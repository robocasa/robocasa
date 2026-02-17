from robocasa.environments.kitchen.kitchen import *


class DeliverStraw(Kitchen):
    """
    Deliver Straw: composite task for Serving Beverages activity.

    Simulates the task of delivering a straw to a glass cup on the dining counter.

    Steps:
        1. Take a straw from the drawer.
        2. Place the straw inside the glass cup on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref(
            "stool",
            dict(id=FixtureType.STOOL),
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Take a straw from the drawer in front and place it inside the glass cup on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="straw",
                obj_groups=("straw"),
                object_scale=[1.5, 1, 2],
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.20),
                    pos=(0.0, -0.25),
                ),
            )
        )
        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups=("glass_cup"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.25, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        straw_in_glass_cup = OU.check_obj_in_receptacle(
            self, "straw", "glass_cup", th=0.5
        )
        gripper_far = OU.gripper_obj_far(self, obj_name="straw")
        return straw_in_glass_cup and gripper_far
