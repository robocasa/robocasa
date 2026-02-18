from robocasa.environments.kitchen.kitchen import *


class PrepareDrinkStation(Kitchen):
    """
    Prepare Drink Station: composite task for Serving Beverages activity.

    Simulates the task of preparing a drink station on the dining counter.

    Steps:
        1. There is a tray on the dining counter.
        2. Grab a cup, mug, and pitcher from the cabinet.
        3. Place the cup and mug on the tray.
        4. Place the pitcher next to the tray within a 0.3 xy distance.
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There is a tray on the dining counter. Grab the cup, mug, and pitcher and "
            "place them on the tray. The pitcher may be placed next to the tray if it cannot fit the tray."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="tray",
                obj_groups=("tray"),
                object_scale=1.2,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=(0.0, -1.0),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="cup",
                obj_groups=("cup"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="mug",
                obj_groups=("mug"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="pitcher",
                obj_groups=("pitcher"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        cup_on_tray = OU.check_obj_in_receptacle(self, "cup", "tray")
        mug_on_tray = OU.check_obj_in_receptacle(self, "mug", "tray")
        pitcher_near_tray = False
        tray_pos = self.sim.data.body_xpos[self.obj_body_id["tray"]][:2]
        pitcher_pos = self.sim.data.body_xpos[self.obj_body_id["pitcher"]][:2]
        dist = np.linalg.norm(tray_pos - pitcher_pos)
        pitcher_near_tray = dist <= 0.3
        gripper_far = (
            OU.gripper_obj_far(self, obj_name="cup")
            and OU.gripper_obj_far(self, obj_name="mug")
            and OU.gripper_obj_far(self, obj_name="pitcher")
        )
        return cup_on_tray and mug_on_tray and pitcher_near_tray and gripper_far
