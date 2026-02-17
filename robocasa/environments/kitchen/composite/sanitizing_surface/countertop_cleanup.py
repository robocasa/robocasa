from robocasa.environments.kitchen.kitchen import *


class CountertopCleanup(Kitchen):
    """
    Countertop Cleanup: composite task for Sanitize Surface activity.

    Simulates the task of cleaning the countertop by organizing items into their proper storage locations.

    Steps:
        1. Place the utensil from the counter into the drawer
        2. Place the receptacle from the counter into the cabinet
        3. Place the bowl of food from the counter into the fridge
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.8, 0.5))
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.cab)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        utensil_name = self.get_obj_lang("utensil")
        receptacle_name = self.get_obj_lang("receptacle")
        food1_name = self.get_obj_lang("food1")
        food2_name = self.get_obj_lang("food2")

        if food1_name == food2_name:
            food_text = f"{food1_name}s"
        else:
            food_text = f"{food1_name} and {food2_name}"

        ep_meta["lang"] = (
            f"Clean up the countertop by placing items away. "
            f"Place the {utensil_name} in the open drawer. "
            f"Place the {receptacle_name} in the open cabinet. "
            f"Place the bowl with the {food_text} in the fridge."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)
        self.fridge.open_door(env=self)
        self.drawer.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="food_bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="receptacle",
                exclude_obj_groups=("pitcher"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="food_bowl",
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil",
                obj_groups=("spoon", "fork", "knife", "wooden_spoon"),
                object_scale=[1, 1, 2.5],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="food_bowl",
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1",
                obj_groups=("vegetable", "cooked_food", "meat"),
                exclude_obj_groups=("kebab_skewer"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="food_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups=("vegetable", "cooked_food", "meat"),
                exclude_obj_groups=("kebab_skewer"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="food_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        utensil_in_drawer = OU.obj_inside_of(
            self, "utensil", self.drawer
        ) and not OU.check_obj_any_counter_contact(self, "utensil")

        receptacle_in_cabinet = OU.obj_inside_of(self, "receptacle", self.cab)
        food1_in_bowl = OU.check_obj_in_receptacle(self, "food1", "food_bowl")
        food2_in_bowl = OU.check_obj_in_receptacle(self, "food2", "food_bowl")
        food_bowl_in_fridge = self.fridge.check_rack_contact(
            self, "food_bowl", compartment="fridge"
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["utensil", "receptacle", "food_bowl"]
        )

        return (
            utensil_in_drawer
            and receptacle_in_cabinet
            and food1_in_bowl
            and food2_in_bowl
            and food_bowl_in_fridge
            and gripper_far
        )
