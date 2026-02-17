from robocasa.environments.kitchen.kitchen import *


class PrepareVeggieDip(Kitchen):
    """
    Prepare Veggie Dip: composite task for Mixing Ingredients activity.
    Simulates the task of preparing a veggie dip by blending vegetables and cream cheese in a blender.
    Steps:
        1) Pick the vegetable and the cream cheese from the fridge
        2) Place them in the blender
        3) Close the blender lid
        4) Turn on the blender
    """

    _BLENDER_PLACEMENT_UPDATE_DICT = {
        "aux_fixture_config": {
            "has_free_joints": True,
            "placement": "parent_region",
            "auxiliary_placement_args": {"ensure_valid_placement": True},
        }
    }
    EXCLUDE_STYLES = Kitchen.PROBLEMATIC_BLENDER_LID_STYLES

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
        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg_lang = self.get_obj_lang("vegetable")
        ep_meta[
            "lang"
        ] = f"Pick the {veg_lang} and the cream cheese from the fridge, place them in the blender, and turn it on."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        self.fridge.open_door(self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cream_cheese",
                obj_groups="cream_cheese_stick",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.35, 0.3),
                    pos=(0.0, -1.0),
                    sample_region_kwargs=dict(compartment="fridge", z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=[
                    "bell_pepper",
                    "carrot",
                    "broccoli",
                    "cucumber",
                    "garlic",
                    "onion",
                    "tomato",
                ],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.2),
                    pos=(0.0, -1.0),
                    sample_region_kwargs=dict(compartment="fridge", z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                obj_groups="all",
                exclude_obj_groups=["vegetable", "cream_cheese_stick"],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.3),
                    pos=(0.0, 0.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):

        objs_inside_blender = OU.obj_inside_of(
            self, "vegetable", self.blender, th=0.01
        ) and OU.obj_inside_of(self, "cream_cheese", self.blender, th=0.01)
        turned_on = self.blender.get_state()["turned_on"]
        return objs_inside_blender and turned_on
