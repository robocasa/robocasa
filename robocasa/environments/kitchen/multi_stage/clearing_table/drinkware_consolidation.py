from robocasa.environments.kitchen.kitchen import *


class DrinkwareConsolidation(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.island = self.register_fixture_ref(
            "island", dict(id=FixtureType.ISLAND)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=self.island),
        )
        self.init_robot_base_pos = self.island

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        objs_lang = self.get_obj_lang("obj_0")
        for i in range(self.num_drinkware):
            objs_lang += f", {self.get_obj_lang(f'obj_{i}')}"
        ep_meta["lang"] = f"pick the {objs_lang} from the island and place it in the open cabinet"
        return ep_meta
    
    def reset(self):
        super().reset()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
    
    def _get_obj_cfgs(self):
        cfgs = []
        self.num_drinkware = self.rng.choice([1, 2, 3])

        for i in range(self.num_drinkware):
            cfgs.append(dict(
                name=f"obj_{i}",
                obj_groups=["drink"],
                graspable=True, washable=True,
                placement=dict(
                    fixture=self.island,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                ),
            ))


        return cfgs

    def _check_success(self):
        objs_in_cab = all([OU.obj_inside_of(self, f"obj_{i}", self.cab) for i in range(self.num_drinkware)])
        gripper_obj_far = all([OU.gripper_obj_far(self, f"obj_{i}") for i in range(self.num_drinkware)])
        return objs_in_cab and gripper_obj_far