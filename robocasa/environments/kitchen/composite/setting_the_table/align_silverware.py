from robocasa.environments.kitchen.kitchen import *


class AlignSilverware(Kitchen):
    """
    Align Silverware Properly: composite task for Setting The Table activity.

    Simulates the task of arranging a fork and a spoon in the correct order next to a plate on the dining table.

    Steps:
        1) Identify the fork and the spoon near the plate.
        2) Place the fork on the left of the plate.
        3) Place the spoon on the right of the plate.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take the fork and place it on the left side of the plate. "
            "Then take the spoon and place it on the right side of the plate. Make sure they're directly next to the plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.15,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fork",
                obj_groups="fork",
                object_scale=[1.15, 1.15, 2.15],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="spoon",
                obj_groups="spoon",
                object_scale=[1.15, 1.15, 2.15],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="fork",
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=(0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fork_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["fork"]])
        spoon_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["spoon"]])
        plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])

        # transform the item positions to the local coordinate system of the dining counter
        fork_x, fork_y = OU.transform_global_to_local(
            fork_pos[0], fork_pos[1], -self.stool.rot + np.pi
        )
        spoon_x, spoon_y = OU.transform_global_to_local(
            spoon_pos[0], spoon_pos[1], -self.stool.rot + np.pi
        )
        plate_x, plate_y = OU.transform_global_to_local(
            plate_pos[0], plate_pos[1], -self.stool.rot + np.pi
        )

        lateral_threshold = 0.25
        forward_threshold = 0.10

        fork_x_dist = abs(fork_x - plate_x)
        fork_y_dist = abs(fork_y - plate_y)
        spoon_x_dist = abs(spoon_x - plate_x)
        spoon_y_dist = abs(spoon_y - plate_y)

        fork_left = (
            (fork_x < plate_x)
            and (fork_x_dist <= lateral_threshold)
            and (fork_y_dist <= forward_threshold)
        )
        spoon_right = (
            (spoon_x > plate_x)
            and (spoon_x_dist <= lateral_threshold)
            and (spoon_y_dist <= forward_threshold)
        )

        fork_on_table = OU.check_obj_fixture_contact(self, "fork", self.dining_counter)
        spoon_on_table = OU.check_obj_fixture_contact(
            self, "spoon", self.dining_counter
        )
        gripper_obj_far_plate = OU.gripper_obj_far(self, obj_name="plate")

        success = (
            fork_left
            and spoon_right
            and fork_on_table
            and spoon_on_table
            and gripper_obj_far_plate
        )

        return success
