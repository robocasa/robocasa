from robocasa.environments.kitchen.kitchen import *


class MakeLoadedPotato(Kitchen):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.6, 0.6), ref=self.microwave)
        )
        self.init_robot_base_pos = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Retrieve the reheated potato from the microwave, then place it on " \
            "the cutting board along with cheese and a bottle of condiment."
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
    
    def _get_obj_cfgs(self):
        cfgs = []
        # Initialize potato in the microwave
        cfgs.append(dict(
            name="potato",
            obj_groups="potato",
            placement=dict(
                fixture=self.microwave,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
                try_to_place_in="bowl",
            ),
        ))

        # Cutting board at the center of the counter
        cfgs.append(dict(
            name="cutting_board",
            obj_groups="cutting_board",
            placement=dict(
                fixture=self.counter,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
                pos=(0, 0),
                rotation=np.pi/2
            ),
        ))

        # Cheese and condiment to be placed on the cutting board
        cfgs.append(dict(
            name="condiment",
            obj_groups="condiment",
            placement=dict(
                fixture=self.counter,
                size=(0.6, 0.5),
                pos=(0, -1)
            ),
        ))
        cfgs.append(dict(
            name="cheese",
            obj_groups="cheese",
            placement=dict(
                fixture=self.counter,
                size=(0.6, 0.5),
                pos=(0, -1)
            ),
        ))
        return cfgs

    def _check_success(self):
        gripper_far = OU.gripper_obj_far(self, "potato") and \
            OU.gripper_obj_far(self, "condiment") and OU.gripper_obj_far(self, "cheese")
        objects_in_place = OU.check_obj_in_receptacle(self, "potato", "cutting_board") and \
            OU.check_obj_in_receptacle(self, "condiment", "cutting_board") and \
            OU.check_obj_in_receptacle(self, "cheese", "cutting_board")
        return gripper_far and objects_in_place
