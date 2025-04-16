from robocasa.environments.kitchen.kitchen import *


class SizeSorting(Kitchen):

    """
    Size Sorting: composite task for Setting The Table activity.

    Simulates the task of stacking objects by size.

    Steps:
        Stack the objects from largest to smallest.
    """

    # exclude layout 9 because objects sometime initilize in corner area which is unreachable
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # sample a large enough counter for multiple stackable categories
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1, 0.4))
        )
        self.init_robot_base_pos = self.counter
        if "object_cfgs" in self._ep_meta:
            object_cfgs = self._ep_meta["object_cfgs"]
            self.num_objs = len(
                [cfg for cfg in object_cfgs if cfg["name"].startswith("obj_")]
            )
        else:
            self.num_objs = self.rng.choice([2, 3])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        stackable_cat = self.get_obj_lang("obj_0")
        ep_meta["lang"] = f"Stack the {stackable_cat}s from largest to smallest."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        stack_cat = self.rng.choice(["cup", "bowl"])
        scale = 0.80
        # pass in object scale to the config to make the objects smaller and thus stackable
        if stack_cat == "cup":
            top_size = (0.6, 0.2)
        elif stack_cat == "bowl":
            top_size = (0.6, 0.4)
        else:
            raise ValueError
        for i in range(self.num_objs):
            obj_cfg = dict(
                name=f"obj_{i}",
                obj_groups=stack_cat,
                object_scale=scale**i,
                placement=dict(
                    fixture=self.counter,
                    ref_obj="obj_0",
                    sample_region_kwargs=dict(top_size=top_size),
                    size=top_size,
                    pos=(None, -1.0),
                    offset=(0.0, 0.0),
                ),
            )
            if i == 0:
                obj_cfg["init_robot_here"] = True
            cfgs.append(obj_cfg)

        return cfgs

    def _check_success(self):

        objs_stacked_inorder = all(
            [
                OU.check_obj_in_receptacle(self, f"obj_{i}", f"obj_{i-1}")
                for i in range(1, self.num_objs)
            ]
        )
        return objs_stacked_inorder and OU.gripper_obj_far(self, "obj_0")
