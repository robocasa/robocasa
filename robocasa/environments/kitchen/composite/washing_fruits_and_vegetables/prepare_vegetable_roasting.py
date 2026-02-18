from robocasa.environments.kitchen.kitchen import *


class PrepareVegetableRoasting(Kitchen):
    """
    Prepare Vegetable Roasting: composite task for Washing Fruits And Vegetables activity.

    Simulates the process of preparing vegetables for roasting by washing it
    in the sink and then placing it on an oven tray.

    Steps:
        Pick the vegetable from the fridge and hold it under the sink faucet to wash it.
        Then place the vegetable on the oven tray next to the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.6))
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"Pick the {vegetable} "
            "from the fridge and hold it under the sink faucet to wash it. "
            "Then place it on the tray next to the sink to prepare for roasting."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.vegetable_washed = False
        self.washed_time = 0
        self.fridge.open_door(self)
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=("vegetable"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(z_range=(1.0, 1.5)),
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray",
                obj_groups="oven_tray",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.7, 0.4),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="existing_vegetable",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    object="tray",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr1",
                fridgable=True,
                exclude_obj_groups=("vegetable", "fruit"),
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.3),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr2",
                fridgable=True,
                exclude_obj_groups=("vegetable", "fruit"),
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.3),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        if self.sink.check_obj_under_water(self, "vegetable"):
            self.washed_time += 1
            self.vegetable_washed = self.washed_time > 10
        else:
            self.washed_time = 0
        vegetables_on_tray = OU.check_obj_in_receptacle(self, "vegetable", "tray")

        return (
            self.vegetable_washed
            and vegetables_on_tray
            and OU.gripper_obj_far(self, "vegetable")
        )
