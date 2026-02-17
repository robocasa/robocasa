from robocasa.environments.kitchen.kitchen import *


class MixCakeFrosting(Kitchen):
    """
    Mix Cake Frosting: composite task for Baking Cookies and Cakes activity.
    Simulates the task of mixing frosting for a cake.
    Steps:
        Add butter and sugar cubes to the stand mixer bowl and turn the knob to begin making frosting
    """

    def __init__(
        self,
        enable_fixtures=None,
        obj_registries=("objaverse", "lightwheel", "aigen"),
        *args,
        **kwargs,
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["stand_mixer"]
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the butter
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(
            enable_fixtures=enable_fixtures,
            obj_registries=obj_registries,
            *args,
            **kwargs,
        )

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stand_mixer = self.register_fixture_ref(
            "stand_mixer", dict(id=FixtureType.STAND_MIXER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, ref=self.stand_mixer)
        )
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Add butter and sugar cubes to the stand mixer bowl and then turn on the stand mixer to begin making frosting."
        return ep_meta

    def _setup_scene(self):
        self.stand_mixer.set_head_pos(self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter_stick",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stand_mixer,
                        loc="left_right",
                    ),
                    size=(0.45, 0.30),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
                object_scale=[0.7, 0.7, 1.0],
            )
        )

        for i in range(2):
            cfgs.append(
                dict(
                    name=f"sugar_{i}",
                    obj_groups="sugar_cube",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.stand_mixer,
                            loc="left_right",
                        ),
                        size=(0.30, 0.15),
                        pos=("ref", "ref"),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        state = self.stand_mixer.get_state(self)
        head_closed = state["head"] < 0.01
        butter_in_bowl = self.stand_mixer.check_item_in_bowl(self, "butter")
        sugars_in_bowl = all(
            [self.stand_mixer.check_item_in_bowl(self, f"sugar_{i}") for i in range(2)]
        )
        stand_mixer_on = self.stand_mixer._speed_dial_knob_value > 0.3
        return sugars_in_bowl and butter_in_bowl and stand_mixer_on and head_closed
