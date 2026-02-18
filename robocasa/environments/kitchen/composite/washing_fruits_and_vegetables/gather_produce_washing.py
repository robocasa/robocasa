from robocasa.environments.kitchen.kitchen import *


class GatherProduceWashing(Kitchen):
    """
    Gather Produce Washing: composite task for Washing Fruits And Vegetables activity.

    Simulates the process of gathering fruits and vegetables from the fridge and
    placing them in a bowl for washing preparation.

    Steps:
        Pick up the fruits/vegetables from the fridge and place them in the bowl
        next to the sink for washing preparation.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        self.fridge.open_door(self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food0_lang = self.get_obj_lang("food0")
        food1_lang = self.get_obj_lang("food1")

        if food0_lang == food1_lang:
            foods_text = f"{food0_lang}s"
        else:
            foods_text = f"{food0_lang} and {food1_lang}"

        ep_meta["lang"] = (
            f"Pick up the {foods_text} from the fridge "
            "and place them in the bowl next to the sink."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="food0",
                obj_groups=("fruit", "vegetable"),
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
                name="distr1",
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
                name=f"food1",
                obj_groups=("fruit", "vegetable"),
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
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_far = all(OU.gripper_obj_far(self, obj) for obj in ["food0", "food1"])
        return (
            OU.check_obj_in_receptacle(self, "food0", "bowl")
            and OU.check_obj_in_receptacle(self, "food1", "bowl")
            and gripper_far
        )
