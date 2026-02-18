from robocasa.environments.kitchen.kitchen import *


class SetupSodaBowl(Kitchen):
    """
    Setup Soda Bowl: composite task for Serving Beverages activity.

    Simulates the task of setting up a bowl with ice on the dining counter and adding sodas from the fridge.

    Steps:
        1. There is a bowl on the dining counter with ice in it.
        2. Open the fridge and grab 2 sodas.
        3. Place both sodas inside the bowl with ice.
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
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Open the fridge, grab two sodas, and place both sodas inside the bowl with ice on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups=("bowl"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice1",
                obj_groups=("ice_cube"),
                object_scale=0.5,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice2",
                obj_groups=("ice_cube"),
                object_scale=0.5,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice3",
                obj_groups=("ice_cube"),
                object_scale=0.5,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="soda1",
                obj_groups=("bottled_drink", "can"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="soda2",
                obj_groups=("bottled_drink", "can"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        soda1_in_bowl = OU.check_obj_in_receptacle(self, "soda1", "ice_bowl")
        soda2_in_bowl = OU.check_obj_in_receptacle(self, "soda2", "ice_bowl")

        if not soda1_in_bowl:
            return False

        if not soda2_in_bowl:
            return False

        bowl_on_dining_counter = OU.check_obj_fixture_contact(
            self, "ice_bowl", self.dining_counter
        )

        gripper_far = (
            OU.gripper_obj_far(self, obj_name="soda1", th=0.15)
            and OU.gripper_obj_far(self, obj_name="soda2", th=0.15)
            and OU.gripper_obj_far(self, obj_name="ice_bowl", th=0.15)
        )

        return (
            soda1_in_bowl and soda2_in_bowl and bowl_on_dining_counter and gripper_far
        )
