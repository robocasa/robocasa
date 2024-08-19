from robocasa.environments.kitchen.kitchen import *


class ServeSteak(Kitchen):
    """
    Serve Steak: composite task for Serving Food activity.

    Simulates the task of serving steak.

    Steps:
        Pick up the pan with the steak in it and place it on the dining table.
        Then, place the steak on the plate.

    Restricted to layouts which have a dining table (long counter area with
    stools).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_pos = self.stove
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the pan with the steak in it and place it on the dining table. Then, place the steak on the plate"
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="steak",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                    container_group="pan",
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=FixtureType.STOOL,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups=("mug", "cup"),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.20),
                    pos=(0.5, 0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        steak_on_plate = OU.check_obj_in_receptacle(self, "obj", "plate")
        pan_on_table = OU.check_obj_fixture_contact(
            self, "obj_container", self.dining_table
        )
        return steak_on_plate and pan_on_table and OU.gripper_obj_far(self)
