from robocasa.environments.kitchen.kitchen import *


class SeparateRawIngredients(Kitchen):
    """
    Separate Raw Ingredients: composite task for Sorting Ingredients activity.

    Simulates the task of separating raw ingredients to avoid cross-contamination
    by placing meat on a cutting board and vegetables in a bowl.

    Steps:
        1. Pick up meat ingredient(s) from the fridge
        2. Place meat on the cutting board
        3. Pick up vegetable ingredient(s) from the fridge
        4. Place vegetable(s) in the bowl
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

        if "refs" in self._ep_meta:
            self.random_ingredient_type = self._ep_meta["refs"][
                "random_ingredient_type"
            ]
        else:
            self.random_ingredient_type = self.rng.choice(["meat", "vegetable"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta[
            "lang"
        ] = f"Place the meat/seafood item(s) on the cutting board and the vegetable item(s) in the bowl to avoid cross-contamination."

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["random_ingredient_type"] = self.random_ingredient_type

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat_ingredient",
                obj_groups="meat",
                exclude_obj_groups=["shrimp"],
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.5, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable_ingredient",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.5, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="random_ingredient",
                obj_groups=self.random_ingredient_type,
                exclude_obj_groups=["shrimp"],
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.5, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.4),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable_bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(0.8, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_on_cutting_board = OU.check_obj_in_receptacle(
            self, "meat_ingredient", "cutting_board"
        )
        vegetable_in_bowl = OU.check_obj_in_receptacle(
            self, "vegetable_ingredient", "vegetable_bowl"
        )

        if self.random_ingredient_type == "meat":
            random_on_cutting_board = OU.check_obj_in_receptacle(
                self, "random_ingredient", "cutting_board"
            )
            random_placement_correct = random_on_cutting_board
        else:
            random_in_bowl = OU.check_obj_in_receptacle(
                self, "random_ingredient", "vegetable_bowl"
            )
            random_placement_correct = random_in_bowl

        gripper_far_meat = OU.gripper_obj_far(self, "meat_ingredient")
        gripper_far_vegetable = OU.gripper_obj_far(self, "vegetable_ingredient")
        gripper_far_random = OU.gripper_obj_far(self, "random_ingredient")

        return (
            meat_on_cutting_board
            and vegetable_in_bowl
            and random_placement_correct
            and gripper_far_meat
            and gripper_far_vegetable
            and gripper_far_random
        )
