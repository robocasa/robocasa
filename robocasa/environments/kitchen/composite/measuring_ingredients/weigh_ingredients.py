from robocasa.environments.kitchen.kitchen import *
from scipy.spatial.transform import Rotation as R


class WeighIngredients(Kitchen):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang("obj")
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                obj_lang=obj_lang
            )
        else:
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} and place it on the digital scale for weighing, and close the cabinet."
        return ep_meta

    def _setup_scene(self):

        self.cab.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("packaged_food"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj_distractor",
                obj_groups=("packaged_food"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="digital_scale",
                obj_groups="digital_scale",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        packaged_food_on_digital_scale = OU.check_obj_in_receptacle(
            self, "obj", "digital_scale"
        )
        obj_upright = OU.check_obj_upright(self, "obj")
        cabinet_closed = self.cab.is_closed(env=self)
        return (
            gripper_obj_far
            and packaged_food_on_digital_scale
            and cabinet_closed
            and obj_upright
        )
