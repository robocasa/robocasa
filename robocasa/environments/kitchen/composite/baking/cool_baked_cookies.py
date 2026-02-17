from robocasa.environments.kitchen.kitchen import *


class CoolBakedCookies(Kitchen):
    """
    Cool Baked Cookies: composite task for Baking Cookies and Cakes activity.

    Simulates the task of finishing baking cookies.
    Steps:
        1. Open the oven door and slide the rack out
        2. Take the tray with baked cookies out of the oven and place it on the counter for cooling
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))

        if self.oven.has_multiple_rack_levels():
            self.rack_level = 1
        else:
            self.rack_level = 0

        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.oven.has_multiple_rack_levels():
            ep_meta["lang"] = (
                f"Open the oven door and slide the top rack out. Then take the tray with baked cookies "
                "out of the oven and place it on the counter for cooling."
            )
        else:
            ep_meta["lang"] = (
                f"Open the oven door and slide the rack out. Then take the tray with baked cookies "
                "out of the oven and place it on the counter for cooling."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.oven.set_temperature(self, 0.5)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="oven_tray",
                obj_groups="oven_tray",
                object_scale=[0.85, 1.0, 1.1],
                placement=dict(
                    fixture=self.oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(1.0, 0.25),
                    pos=(0, -1.0),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cookie0",
                obj_groups=("cookie_dough_ball"),
                object_scale=[1.75, 1.75, 0.75],
                placement=dict(
                    object="oven_tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cookie1",
                obj_groups=("cookie_dough_ball"),
                object_scale=[1.75, 1.75, 0.75],
                placement=dict(
                    object="oven_tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cookies_on_tray = all(
            [
                OU.check_obj_in_receptacle(self, f"cookie{i}", "oven_tray")
                for i in range(2)
            ]
        )

        tray_on_counter = OU.check_obj_any_counter_contact(self, "oven_tray")
        gripper_away = OU.gripper_obj_far(self, "oven_tray")

        return cookies_on_tray and tray_on_counter and gripper_away
