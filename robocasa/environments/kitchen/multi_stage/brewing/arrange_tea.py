from robocasa.environments.kitchen.kitchen import *


class ArrangeTea(Kitchen):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.DOOR_TOP_HINGE_DOUBLE))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4)))
        self.init_robot_base_pos = self.cab
    
    def _reset_internal(self):
        super()._reset_internal()
        self.cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
    
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the kettle from the counter and place it on the tray. Then pick the mug from the cabinet and place it on the tray. Then close the cabinet doors."
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj",
            obj_groups=("mug"),
            graspable=True,
            placement=dict(
                fixture=self.cab,
                size=(0.50, 0.20),
                pos=(0, -1.0),
            ),
        ))

        cfgs.append(dict(
            name="obj2",
            obj_groups=("kettle"),
            graspable=True,
            placement=dict(
                fixture=self.counter,
                size=(0.5, 0.5),
                pos=("ref", -1.0),
                sample_region_kwargs=dict(
                    ref=self.cab,
                    top_size=(0.6, 0.4)
                ),
                offset=(0.1, 0.0),
            ),
        ))

        cfgs.append(dict(
            name="container",
            obj_groups=("tray"),
            placement=dict(
                fixture=self.counter,
                size=(0.7, 0.7),
                pos=("ref", -0.6),
                offset=(-0.1, 0.0),
                
                sample_region_kwargs=dict(
                    ref=self.cab,  
                    top_size=(0.6,0.4)
                ),
            ),
        ))

        return cfgs
    
    def _check_door_closed(self):
        door_state = self.cab.get_door_state(env=self)
        
        success = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                success = False
                break

        return success

    
    def _check_success(self):
        obj1_container_contact = OU.check_obj_in_receptacle(self, "obj", "container") 
        obj2_container_contact = OU.check_obj_in_receptacle(self, "obj2", "container")
        door_closed = self._check_door_closed()
        gripper_obj_far = OU.gripper_obj_far(self) # no need to check all gripper objs far bc all objs in the same place


        return obj1_container_contact  and obj2_container_contact and gripper_obj_far and door_closed


    