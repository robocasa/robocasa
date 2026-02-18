from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object


class ChooseRipeFruit(Kitchen):
    """
    ChooseRipeFruit: composite task for the making juice activity.
    Simulates the task of choosing a ripe fruit from two fruits (one rotten and one non-rotten) and putting it in the blender.
    Steps:
        1. Pick up the non-rotten fruit from the counter.
        2. Put the non-rotten fruit in the blender.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender, size=(0.35, 0.35))
        )

        self.init_robot_base_ref = self.blender

        if "refs" in self._ep_meta:
            self.chosen_inst = self._ep_meta["refs"]["chosen_inst"]
        else:
            self.chosen_inst = sample_kitchen_object(
                obj_registries=self.obj_registries,
                groups="fruit",
                rng=self.rng,
                graspable=True,
                split=self.obj_instance_split,
            )[0]["mjcf_path"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit_lang = OU.get_obj_lang(self, "fruit1")
        ep_meta["lang"] = f"Put the non-rotten {fruit_lang} in the blender."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["chosen_inst"] = self.chosen_inst

        return ep_meta

    def _update_fruit_texture(self):
        """
        Updates the texture of the fruit object to be rotten.
        """

        # arbitrarily choose fruit1 to be the ripe fruit and fruit0 to be rotten.
        # Since locations are randomized it doesnt matter
        fruit0 = self.objects["fruit0"]
        for geom in fruit0.visual_geoms:
            name = geom
            # brown color for rotten fruit
            rgb = [0.4, 0.26, 0.14]
            geom_id = self.sim.model.geom_name2id(name)
            self.sim.model.geom_rgba[geom_id, :3] = rgb
            prev_alpha = self.sim.model.geom_rgba[geom_id, 3]
            self.sim.model.geom_rgba[geom_id, 3] = min(prev_alpha * 2, 1)

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self._update_fruit_texture()

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"fruit{i}",
                    obj_groups=self.chosen_inst,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.35, 0.15),
                        pos=("ref", -1.0),
                        sample_region_kwargs=dict(loc="left_right", ref=self.blender),
                    ),
                )
            )
        return cfgs

    def _check_success(self):

        ripe_fruit_in_blender = OU.obj_inside_of(self, f"fruit1", self.blender, th=0.01)
        gripper_fruit_far = OU.gripper_obj_far(self, "fruit1")
        rotten_fruit_not_in_blender = not OU.obj_inside_of(
            self, f"fruit0", self.blender, th=0.01
        )
        return (
            ripe_fruit_in_blender and gripper_fruit_far and rotten_fruit_not_in_blender
        )
