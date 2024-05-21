from robocasa.environments.kitchen.kitchen import *


class SizeSorting(Kitchen):

    def __init__(self,*args, **kwargs):

        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, size=(1, 0.4)))  
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        stackable_cat = self.get_obj_lang("obj_0")
        ep_meta["lang"] = f"Stack the {stackable_cat}s from largest to smallest"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        self.objs = random.choice([2,3,4])
        stack_cat = random.choice(["cup", "bowl"])
        scale=0.8
        for i in range(self.objs):
            cfgs.append(dict(
            name=f"obj_{i}",
            obj_groups=stack_cat,
            object_scale=scale **i,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    top_size=(0.6, 0.4)
                    ),
                size=(0.6, 0.4),
                pos=(0, -1.0),
                offset=(i*0.1, 0),
                ),
            ))

        return cfgs

    def _check_success(self):

        objs_stacked_inorder = all([OU.check_obj_in_receptacle(self, f"obj_{i}", f"obj_{i-1}") for i in range(1,self.objs)])
        return objs_stacked_inorder and OU.gripper_obj_far(self, "obj_0")