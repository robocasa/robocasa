from robocasa.environments.kitchen.kitchen import *


class BlendVegetableSauce(Kitchen):
    """
    Blend Vegetable Sauce: composite task for Mixing Ingredients activity.
    Simulates the task of blending boiled vegetables to make a vegetable sauce.
    Steps:
        1) Turn off the stove burner for the saucepan
        2) Place boiled vegetable in the blender
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

        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )

        self.init_robot_base_ref = self.stove
        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob_id"]
        else:
            valid_knobs = []
            for knob, joint in self.stove.knob_joints.items():
                if joint is not None and not knob.startswith("rear"):
                    valid_knobs.append(knob)
            self.knob = self.rng.choice(valid_knobs)

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="on", knob=self.knob, env=self, rng=self.rng)
        OU.add_obj_liquid_site(self, "saucepan", [0.5, 0.6, 1.0, 0.3])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg_lang = self.get_obj_lang("vegetable")
        ep_meta[
            "lang"
        ] = f"The {veg_lang} is done being boiled. Turn off the stove burner for the saucepan. Place the {veg_lang} in the blender and turn on the blender to make a vegetable sauce."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob_id"] = self.knob

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan",
                object_scale=[1.35, 1.35, 0.85],
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
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
                    "onion",
                    "tomato",
                ],
                graspable=True,
                placement=dict(
                    object="saucepan",
                    size=(1.0, 0.6),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sauce",
                obj_groups=["mayonnaise", "ketchup", "vinegar"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        stove_off = not self.stove.is_burner_on(env=self, burner_loc=self.knob)
        vegetable_in_blender = OU.obj_inside_of(
            self, "vegetable", self.blender, th=0.01
        )
        blender_state = self.blender.get_state()
        return vegetable_in_blender and blender_state["turned_on"] and stove_off
