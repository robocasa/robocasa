from robocasa.environments.kitchen.kitchen import *


class SeasoningSteak(Kitchen):
    """
    Seasoning Steak: composite task for Seasoning Food.

    Simulates the task of preparing salt or pepper to season a steak on a plate.

    Steps:
        Retrieve shaker from cabinet and place it beside the steak on a plate.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Retrieve the shaker from the cabinet and place it next the steak on a plate on the dining table."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="shaker",
                obj_groups=("shaker", "salt_and_pepper_shaker"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.30, 0.20),
                    pos=(0.0, -0.90),
                ),
            )
        )

        cfgs.append(
            dict(
                name="steak",
                obj_groups="steak",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        # this is necessary to detect if the dining counter is rotated around the global/absolute coordinate system
        _, rel_rot = OU.get_rel_transform(self.dining_counter, self.stool)
        relative_yaw = np.arctan2(rel_rot[1, 0], rel_rot[0, 0])

        steak_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["steak"]])
        shaker_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["shaker"]])

        forward_threshold = 0.10
        side_threshold = 0.25

        steak_x, steak_y = OU.transform_global_to_local(
            steak_pos[0], steak_pos[1], (np.pi + relative_yaw) - self.dining_counter.rot
        )
        shaker_x, shaker_y = OU.transform_global_to_local(
            shaker_pos[0],
            shaker_pos[1],
            (np.pi + relative_yaw) - self.dining_counter.rot,
        )

        shaker_x_dist = abs(shaker_x - steak_x)
        shaker_y_dist = abs(shaker_y - steak_y)

        shaker_near = (shaker_y_dist <= forward_threshold) and (
            shaker_x_dist <= side_threshold
        )
        shaker_on_counter = OU.check_obj_fixture_contact(
            self, "shaker", self.dining_counter
        )

        gripper_obj_far_steak = OU.gripper_obj_far(self, obj_name="steak")
        return shaker_near and shaker_on_counter and gripper_obj_far_steak
