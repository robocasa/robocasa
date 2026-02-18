from robocasa.environments.kitchen.kitchen import *


class AddSugarCubes(Kitchen):
    """
    Add Sugar Cube to Cake: composite task for Garnishing Dishes activity.

    Simulates the task of garnishing a cake with two sugar cubes.

    Steps:
        Take two sugar cubes from the cabinet and place them on a plate with cake
        on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.stool

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Take the two sugar cubes and place them on the plate with the cake."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cake",
                obj_groups="cake",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.35, 0.25),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar_cube_1",
                obj_groups="sugar_cube",
                object_scale=[1, 1, 1.5],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.5, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar_cube_2",
                obj_groups="sugar_cube",
                object_scale=[1, 1, 1.5],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.5, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        sugar_cube_1_on_plate = OU.check_obj_in_receptacle(
            self, "sugar_cube_1", "cake_container"
        )
        sugar_cube_2_on_plate = OU.check_obj_in_receptacle(
            self, "sugar_cube_2", "cake_container"
        )
        cake_on_plate = OU.check_obj_in_receptacle(self, "cake", "cake_container")
        plate_on_table = OU.check_obj_fixture_contact(
            self, "cake_container", self.dining_counter
        )
        gripper_far_1 = OU.gripper_obj_far(self, "sugar_cube_1", th=0.15)
        gripper_far_2 = OU.gripper_obj_far(self, "sugar_cube_2", th=0.15)
        return (
            sugar_cube_1_on_plate
            and sugar_cube_2_on_plate
            and cake_on_plate
            and plate_on_table
            and gripper_far_1
            and gripper_far_2
        )
