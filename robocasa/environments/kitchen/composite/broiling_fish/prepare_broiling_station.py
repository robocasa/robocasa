from robocasa.environments.kitchen.kitchen import *


class PrepareBroilingStation(Kitchen):
    """
    Prepare Broiling Station: composite task for Broiling Fish activity.

    Simulates the process of setting up cooking utensils near the oven for broiling.

    Steps:
        Take the tong from the counter and place it on the plate near the oven.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter_oven = self.register_fixture_ref(
            "counter_oven", dict(id=FixtureType.COUNTER, ref=self.oven)
        )
        self.counter_items = self.register_fixture_ref(
            "counter_items",
            dict(id=FixtureType.COUNTER_NON_CORNER, full_depth_region=True),
        )
        self.init_robot_base_ref = self.counter_items

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the tongs from the counter and place them near the plate next to the oven."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.oven.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tongs",
                obj_groups="tongs",
                object_scale=[1, 1, 1.5],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter_items,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(0.2, 0.4),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter_oven,
                    sample_region_kwargs=dict(ref=self.oven),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_far = OU.gripper_obj_far(self, obj_name="tongs")

        tong_on_counter = OU.check_obj_any_counter_contact(self, obj_name="tongs")
        plate_on_counter = OU.check_obj_any_counter_contact(self, obj_name="plate")

        tongs_pos = self.sim.data.body_xpos[self.obj_body_id["tongs"]][:2]
        plate_pos = self.sim.data.body_xpos[self.obj_body_id["plate"]][:2]
        dist_xy = np.linalg.norm(np.array(tongs_pos) - np.array(plate_pos))

        tongs_near_plate = dist_xy < 0.30

        return gripper_far and tongs_near_plate and tong_on_counter and plate_on_counter
