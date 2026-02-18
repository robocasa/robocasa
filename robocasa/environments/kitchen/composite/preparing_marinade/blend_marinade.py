from robocasa.environments.kitchen.kitchen import *


class BlendMarinade(Kitchen):
    """
    Blend Marinade: composite task for Preparing Marinade activity.

    Simulates the task of creating a marinade by blending vegetables in a blender.

    Steps:
        1. Retrieve two marinade vegetables from the fridge
        2. Place them in the blender
        3. Close the blender lid
        4. Turn on the blender by pressing the power button
    """

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
        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        veg1_lang = self.get_obj_lang("marinade_veg1")
        veg2_lang = self.get_obj_lang("marinade_veg2")

        if veg1_lang == veg2_lang:
            veg_text = f"{veg1_lang}s"
        else:
            veg_text = f"{veg1_lang} and {veg2_lang}"

        ep_meta["lang"] = (
            f"Grab the {veg_text} from the fridge, "
            f"place them in the blender, close the lid, and turn on the blender to create a marinade."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="marinade_veg1",
                obj_groups=("garlic", "onion", "lemon", "lemon_wedge", "lime"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.15),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="marinade_veg2",
                obj_groups=("garlic", "onion", "lemon", "lemon_wedge", "lime"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.15),
                    pos=(0.3, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg1_in_blender = OU.obj_inside_of(self, "marinade_veg1", self.blender, th=0.01)
        veg2_in_blender = OU.obj_inside_of(self, "marinade_veg2", self.blender, th=0.01)

        blender_on = self.blender.get_state()["turned_on"]

        return veg1_in_blender and veg2_in_blender and blender_on
