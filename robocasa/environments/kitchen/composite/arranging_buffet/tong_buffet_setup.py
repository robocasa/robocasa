from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import *


class TongBuffetSetup(ManipulateDrawer):
    """
    Tong Buffet Setup: composite task for Arranging Buffet.

    Simulates taking tongs from the drawer and placing it next to a tray on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=FixtureType.DRAWER))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food = self.get_obj_lang("food")

        ep_meta["lang"] = (
            f"Take the tongs from the drawer on the {self.drawer_side} and place them on the dining counter "
            f"next to the tray containing the {food}."
        )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tongs",
                obj_groups="tongs",
                object_scale=(1.0, 1.0, 2.5),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.5),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food",
                obj_groups="cooked_food",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2),
                    try_to_place_in="tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_spoon",
                obj_groups="spoon",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 1.0),
                    rotation=(np.pi),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_fork",
                obj_groups="fork",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    reuse_region_from="distractor_spoon",
                    size=(1.0, 1.0),
                    rotation=(np.pi),
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
        tongs_on_counter = OU.check_obj_fixture_contact(
            self, "tongs", self.dining_counter
        )
        tray_on_counter = OU.check_obj_fixture_contact(
            self, "food_container", self.dining_counter
        )

        if tongs_on_counter and tray_on_counter:
            tongs_pos = self.sim.data.body_xpos[self.obj_body_id["tongs"]][:2]
            tray_pos = self.sim.data.body_xpos[self.obj_body_id["food_container"]][:2]
            distance = np.linalg.norm(tongs_pos - tray_pos)
            tongs_near_tray = distance <= 0.5
        else:
            tongs_near_tray = False

        gripper_far = OU.gripper_obj_far(
            self, obj_name="tongs", th=0.15
        ) and OU.gripper_obj_far(self, obj_name="food_container", th=0.15)

        return tongs_on_counter and tray_on_counter and tongs_near_tray and gripper_far
