from robocasa.environments.kitchen.kitchen import *


class ArrangeCuttingFruits(Kitchen):
    """
    Arrange Cutting Fruits: composite task for Chopping Food activity.

    Simulates the task of selecting fruits from the fridge and placing them on the cutting board
    while leaving non-fruit items in the fridge.

    Steps:
        1. Select and place the 2 fruits from the fridge on the cutting board
        2. Leave the 2 non-fruit items in the fridge
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            f"Place the fruits in the fridge on the cutting board by the sink to make a fruit bowl. "
            f"Leave the non-fruit items in the fridge."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.45, 0.55)
                    ),
                    size=(0.35, 0.45),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.7),
                    pos=(0, 0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups=("fruit"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.5, 0.25),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups=("fruit"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.5, 0.25),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="non_fruit1",
                exclude_obj_groups="fruit",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.8, 0.25),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="non_fruit2",
                exclude_obj_groups="fruit",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.8, 0.25),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit1_on_board = OU.check_obj_in_receptacle(self, "fruit1", "cutting_board")
        fruit2_on_board = OU.check_obj_in_receptacle(self, "fruit2", "cutting_board")

        non_fruit1_in_fridge = self.fridge.check_rack_contact(
            self, "non_fruit1", compartment="fridge"
        )
        non_fruit2_in_fridge = self.fridge.check_rack_contact(
            self, "non_fruit2", compartment="fridge"
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["fruit1", "fruit2", "non_fruit1", "non_fruit2"]
        )

        return all(
            [
                fruit1_on_board,
                fruit2_on_board,
                non_fruit1_in_fridge,
                non_fruit2_in_fridge,
                gripper_far,
            ]
        )
