from robocasa.environments.kitchen.kitchen import *


class PlaceIceInCup(Kitchen):
    """
    Place Ice in Cup: composite task for Adding Ice to Beverages activity.

    Simulates placing 2 ice cubes into 1 glass cup.

    Steps:
        1. Pick up an ice cube from the bowl.
        2. Place it in the glass cup.
        3. Repeat for the second ice cube.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There are two ice cubes in the ice bowl. "
            "Place both ice cubes in the glass cup."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups="bowl",
                object_scale=[1.1, 1.1, 0.75],
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_cube1",
                obj_groups="ice_cube",
                object_scale=1.0,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="ice_cube2",
                obj_groups="ice_cube",
                object_scale=1.0,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups="glass_cup",
                object_scale=[1.35, 1.35, 1.2],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.25, 0.25),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pitcher",
                obj_groups="pitcher",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(1.0, 0.5),
                    pos=(0, -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        ice1_in_cup = OU.check_obj_in_receptacle(self, "ice_cube1", "glass_cup", th=0.5)
        ice2_in_cup = OU.check_obj_in_receptacle(self, "ice_cube2", "glass_cup", th=0.5)
        gripper_far_ice1 = OU.gripper_obj_far(self, "ice_cube1", th=0.15)
        gripper_far_ice2 = OU.gripper_obj_far(self, "ice_cube2", th=0.15)

        ice1_ice2_contact = self.check_contact(
            self.objects["ice_cube1"], self.objects["ice_cube2"]
        )

        success = (
            (
                (ice1_in_cup and (ice2_in_cup or ice1_ice2_contact))
                or (ice2_in_cup and (ice1_in_cup or ice1_ice2_contact))
            )
            and gripper_far_ice1
            and gripper_far_ice2
        )
        return success
