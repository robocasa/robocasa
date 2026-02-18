from robocasa.environments.kitchen.kitchen import *


class PlateStoreDinner(Kitchen):
    """
    Plate Store Dinner: composite task for Plating Food activity.
    Simulates the task of plating and storing dinner.

    Steps:
        1. Place the meat on the plate.
        2. Place the other meat in the tupperware.
        3. Store the tupperware in the fridge.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Place one of the steaks on the plate and the other in the bowl. "
            f"Then place the bowl in the fridge."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat1",
                obj_groups="steak",
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.2, 0.2),
                    pos=("ref", -1.0),
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat2",
                obj_groups="steak",
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.2, 0.2),
                    pos=("ref", -1.0),
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.60, 0.45),
                    pos=("ref", -1.0),
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
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.60, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat1_on_plate = OU.check_obj_in_receptacle(self, "meat1", "plate")

        meat2_on_plate = OU.check_obj_in_receptacle(self, "meat2", "plate")

        meat1_in_tupper = OU.check_obj_in_receptacle(self, "meat1", "bowl")

        meat2_in_tupper = OU.check_obj_in_receptacle(self, "meat2", "bowl")

        gripper_far = OU.gripper_obj_far(self, "meat1") and OU.gripper_obj_far(
            self, "meat2"
        )

        meat2_in_fridge = OU.obj_inside_of(self, "meat2", self.fridge)

        meat1_in_fridge = OU.obj_inside_of(self, "meat1", self.fridge)

        return (
            meat1_on_plate and meat2_in_tupper and meat2_in_fridge and gripper_far
        ) or (meat2_on_plate and meat1_in_tupper and meat1_in_fridge and gripper_far)
