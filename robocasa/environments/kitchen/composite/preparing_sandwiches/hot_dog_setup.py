from robocasa.environments.kitchen.kitchen import *


class HotDogSetup(Kitchen):
    """
    Prepare a Hot Dog Setup: composite task for Preparing Sandwiches.

    Simulates gathering the necessary ingredients for a hot dog and placing them on the dining table.

    Steps:
        1) Pick up the hot dog bun from the counter and place it on the plate.
        2) Pick up the sausage from the fridge and place it on the plate.
        3) Pick up the condiment bottle and place it next to the plate.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("aigen", "objaverse", "lightwheel"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.dining_table = self.register_fixture_ref(
            "dining_table", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        condiment_lang = self.get_obj_lang("condiment")
        ep_meta["lang"] = (
            f"Move the hot dog bun to the plate on the dining table and place the {condiment_lang} next to that plate. "
            "Then navigate to the fridge and take the sausage and place it on the plate as well."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="hotdog_bun",
                obj_groups="hotdog_bun",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.45, 0.40),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="sausage",
                obj_groups="sausage",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.45, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups=("ketchup", "mustard", "condiment_bottle"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.50, 0.20),
                    pos=(-0.3, -0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.35,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Ensure the hot dog ingredients are properly set up:
        1. The hot dog bun and sausage are on the plate.
        2. The plate is placed on the dining table.
        3. The condiment bottle is placed near the plate (within a certain XY distance).
        4. The gripper is far from all objects.
        """

        bun_on_plate = OU.check_obj_in_receptacle(self, "hotdog_bun", "plate")
        sausage_on_plate = OU.check_obj_in_receptacle(self, "sausage", "plate")

        plate_on_table = OU.check_obj_fixture_contact(self, "plate", self.dining_table)

        plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])
        condiment_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["condiment"]])

        max_xy_distance = 0.35
        condiment_near_plate = (
            np.linalg.norm(plate_pos[:2] - condiment_pos[:2]) < max_xy_distance
        )
        condiment_on_table = OU.check_obj_fixture_contact(
            self, "condiment", self.dining_table
        )

        gripper_far_from_objects = (
            OU.gripper_obj_far(self, "hotdog_bun")
            and OU.gripper_obj_far(self, "sausage")
            and OU.gripper_obj_far(self, "condiment")
            and OU.gripper_obj_far(self, "plate")
        )

        return (
            bun_on_plate
            and sausage_on_plate
            and plate_on_table
            and condiment_near_plate
            and condiment_on_table
            and gripper_far_from_objects
        )
