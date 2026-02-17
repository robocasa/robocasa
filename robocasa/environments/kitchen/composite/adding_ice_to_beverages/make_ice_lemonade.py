from robocasa.environments.kitchen.kitchen import *


class MakeIceLemonade(Kitchen):
    """
    Prepare Ice Lemonade: composite task for Adding Ice to Beverages activity.

    Simulates grabbing a lemon wedge from the fridge and one ice cube from an ice bowl and placing one cube in a glass cup.

    Steps:
        1. Pick up a lemon wedge from the fridge and place it in the glass cup.
        2. Pick up an ice cube from the ice bowl and place it in the glass cup.
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

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Grab a lemon wedge from the fridge and one ice cube from the ice bowl,"
                " and put them in the glass of lemonade."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "glass_cup", [0.95, 0.95, 0.6, 0.25])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="lemon_wedge",
                obj_groups="lemon_wedge",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.25, 0.20),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                fridgable=True,
                exclude_obj_groups="lemon_wedge",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0, 1.0),
                    sample_region_kwargs=dict(
                        z_range=(0.5, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups="bowl",
                object_scale=[1.1, 1.1, 0.75],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_cube1",
                obj_groups="ice_cube",
                object_scale=0.9,
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
                object_scale=0.9,
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
                object_scale=[1.25, 1.25, 1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.5, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        lemon_in_cup = OU.check_obj_in_receptacle(
            self, "lemon_wedge", "glass_cup", th=0.5
        )
        ice1_in_cup = OU.check_obj_in_receptacle(self, "ice_cube1", "glass_cup", th=0.5)
        ice2_in_cup = OU.check_obj_in_receptacle(self, "ice_cube2", "glass_cup", th=0.5)
        ice_in_cup = ice1_in_cup or ice2_in_cup
        gripper_far_lemon = OU.gripper_obj_far(self, "lemon_wedge", th=0.15)
        gripper_far_ice = all(
            [
                OU.gripper_obj_far(self, "ice_cube1", th=0.15),
                OU.gripper_obj_far(self, "ice_cube2", th=0.15),
            ]
        )

        success = lemon_in_cup and ice_in_cup and gripper_far_lemon and gripper_far_ice
        return success
