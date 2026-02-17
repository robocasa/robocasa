from robocasa.environments.kitchen.kitchen import *


class MakeBananaMilkshake(Kitchen):
    """
    Make Banana Milkshake: composite task for Mixing and Blending activity.

    Simulates the task of preparing ingredients for a banana milkshake by placing
    a banana in the blender jug and positioning milk and honey next to the blender.

    Steps:
        1. Place honey next to the blender.
        2. Place a banana in the blender jug
        3. Place milk next to the blender
    """

    # update yaml config to place the blender lid next to the blender instead of on top of it
    _BLENDER_PLACEMENT_UPDATE_DICT = {
        "aux_fixture_config": {
            "has_free_joints": True,
            "placement": "parent_region",
            "auxiliary_placement_args": {"ensure_valid_placement": True},
        }
    }

    def __init__(
        self, enable_fixtures=None, update_fxtr_cfg_dict=None, *args, **kwargs
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]

        update_fxtr_cfg_dict = update_fxtr_cfg_dict or {}
        update_fxtr_cfg_dict["blender"] = self._BLENDER_PLACEMENT_UPDATE_DICT
        super().__init__(
            enable_fixtures=enable_fixtures,
            update_fxtr_cfg_dict=update_fxtr_cfg_dict,
            *args,
            **kwargs,
        )

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )

        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Grab the honey bottle and place it next to the blender. Then grab the milk and banana from the fridge, "
            "and place the milk next to the blender while placing the banana in the blender jug to prepare making a banana milkshake."
        )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)
        self.cabinet.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="banana",
                obj_groups="banana",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.5, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1, 1.5),
                    ),
                    size=(0.5, 0.15),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                exclude_obj_groups=("milk", "honey_bottle", "banana"),
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.3),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="honey",
                obj_groups="honey_bottle",
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        banana_in_blender = OU.obj_inside_of(self, "banana", self.blender, th=0.01)

        milk_distance = OU.obj_fixture_bbox_min_dist(self, "milk", self.blender)
        honey_distance = OU.obj_fixture_bbox_min_dist(self, "honey", self.blender)

        honey_near_blender = honey_distance <= 0.3
        milk_near_blender = milk_distance <= 0.3
        honey_on_counter = OU.check_obj_any_counter_contact(self, "honey")
        milk_on_counter = OU.check_obj_any_counter_contact(self, "milk")

        gripper_far = all(
            [
                OU.gripper_obj_far(self, obj_name="banana"),
                OU.gripper_obj_far(self, obj_name="milk"),
                OU.gripper_obj_far(self, obj_name="honey"),
            ]
        )

        return all(
            [
                banana_in_blender,
                milk_near_blender,
                honey_near_blender,
                honey_on_counter,
                milk_on_counter,
                gripper_far,
            ]
        )
