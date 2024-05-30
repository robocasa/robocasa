from robocasa.environments.kitchen.kitchen import *


class BowlAndCup(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.island = self.register_fixture_ref(
            "island", dict(id=FixtureType.ISLAND)
        )
        
        self.init_robot_base_pos = self.island

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"place the cup inside the bowl on the island and move it to any counter" 
        return ep_meta
    
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name=f"cup",
            obj_groups=["cup"],
            graspable=True, washable=True,
            placement=dict(
                fixture=self.island,
                size=(0.30, 0.40),
                pos=(0, -1.0),
            ),
        ))

        cfgs.append(dict(
            name=f"bowl",
            obj_groups=["bowl"],
            graspable=True, washable=True,
            placement=dict(
                fixture=self.island,
                size=(0.30, 0.40),
                pos=(0, -1.0),
            ),
        ))




        return cfgs

    def _check_success(self):
        cup_in_bowl = OU.check_obj_in_receptacle(self, "cup", "bowl")
        bowl_on_counter = any([OU.check_obj_fixture_contact(self, "bowl", fxtr) for (_, fxtr) in self.fixtures.items() if isinstance(fxtr, Counter) and fxtr != self.island] )
        return cup_in_bowl and bowl_on_counter and OU.gripper_obj_far(self, "bowl")