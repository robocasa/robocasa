from robocasa.environments.kitchen.kitchen import *


class PackFruitContainer(Kitchen):
    """
    Pack Fruit Container: composite task for Packing Lunches activity.

    Simulates the task of packing two fruits from the fridge into a tupperware on the counter.

    Steps:
        1. Pick two fruits from the fridge.
        2. Place them inside a tupperware on the counter.
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

    def _setup_scene(self):
        self.fridge.open_door(self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit0_lang = self.get_obj_lang("fruit0")
        fruit1_lang = self.get_obj_lang("fruit1")
        container_lang = self.get_obj_lang("tupperware")

        if fruit0_lang == fruit1_lang:
            ep_meta[
                "lang"
            ] = f"Pick up both {fruit0_lang}s from the fridge and place them inside the {container_lang} on the counter."
        else:
            ep_meta[
                "lang"
            ] = f"Pick the {fruit0_lang} and {fruit1_lang} from the fridge and place them inside the {container_lang} on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="fruit0",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=(0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return (
            OU.check_obj_in_receptacle(self, "fruit0", "tupperware")
            and OU.check_obj_in_receptacle(self, "fruit1", "tupperware")
            and OU.gripper_obj_far(self, "fruit0")
            and OU.gripper_obj_far(self, "fruit1")
        )
