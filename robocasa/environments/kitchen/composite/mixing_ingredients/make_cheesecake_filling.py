from robocasa.environments.kitchen.kitchen import *


class MakeCheesecakeFilling(Kitchen):
    """
    Make Cheesecake Filling: composite task for Mixing Ingredients activity.

    Simulates the task of making cheesecake filling using butter, sugar, and cream cheese in a stand mixer.

    Steps:
        1. Add butter stick, sugar cube, and cream cheese stick to the stand mixer bowl
        2. Turn on the stand mixer to begin making cheesecake filling
    """

    EXCLUDE_LAYOUTS = [57]

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
        # make sure to use aigen objects to access the butter and cream cheese
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
            "counter", dict(id=FixtureType.COUNTER, ref=self.stand_mixer)
        )
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Add the butter stick, sugar cube, and cream cheese stick to the stand mixer bowl and then turn the speed knob to begin making cheesecake filling."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stand_mixer.set_head_pos(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.1,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stand_mixer,
                        loc="left_right",
                    ),
                    size=(0.45, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter_stick",
                object_scale=[0.7, 0.7, 1.0],
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cream_cheese",
                obj_groups="cream_cheese_stick",
                object_scale=[0.7, 0.7, 1.75],
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar",
                obj_groups="sugar_cube",
                object_scale=1.1,
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        state = self.stand_mixer.get_state(self)
        head_closed = state["head"] < 0.05

        butter_in_bowl = self.stand_mixer.check_item_in_bowl(self, "butter")
        sugar_in_bowl = self.stand_mixer.check_item_in_bowl(self, "sugar")
        cream_cheese_in_bowl = self.stand_mixer.check_item_in_bowl(self, "cream_cheese")
        stand_mixer_on = self.stand_mixer._speed_dial_knob_value > 0.2

        return (
            butter_in_bowl
            and sugar_in_bowl
            and cream_cheese_in_bowl
            and stand_mixer_on
            and head_closed
        )
