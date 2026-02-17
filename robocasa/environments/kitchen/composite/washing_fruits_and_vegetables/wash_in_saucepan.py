from robocasa.environments.kitchen.kitchen import *


class WashInSaucepan(Kitchen):
    """
    Wash In Saucepan: composite task for Washing Fruits And Vegetables activity.

    Simulates the process of taking vegetables from the fridge and placing them
    in a saucepan with water next to the sink for washing.

    Steps:
        Pick up 2 vegetables from the fridge and place them in the saucepan
        with water next to the sink for washing.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg0_lang = self.get_obj_lang("vegetable0")
        veg1_lang = self.get_obj_lang("vegetable1")

        if veg0_lang == veg1_lang:
            vegetables_text = f"{veg0_lang}s"
        else:
            vegetables_text = f"{veg0_lang} and {veg1_lang}"

        ep_meta["lang"] = (
            f"Pick up the {vegetables_text} from the fridge "
            "and place them in the saucepan with water next to the sink for soak washing."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)
        OU.add_obj_liquid_site(self, "saucepan", [0.5, 0.6, 1.0, 0.3])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable0",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("fruit", "vegetable"),
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan",
                object_scale=[1.2, 1.2, 1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.55, 0.35),
                    pos=("ref", -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        vegetables_in_saucepan = OU.check_obj_in_receptacle(
            self, "vegetable0", "saucepan"
        ) and OU.check_obj_in_receptacle(self, "vegetable1", "saucepan")
        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["vegetable0", "vegetable1"]
        )

        return gripper_far and vegetables_in_saucepan
