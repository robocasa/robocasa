from robocasa.environments.kitchen.kitchen import *


class RestockBowls(Kitchen):
    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4)))
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")

        ep_meta["lang"] = f"Open the cabinet. Pick the {obj_name_1} and the {obj_name_2} from the counter and place it in the cabinet directly in front. Then close the cabinet."

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []        
        cfgs.append(dict(
            name="obj1",
            obj_groups="bowl",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.cab,
                    top_size=(0.6, 0.4)
                ),
                size=(0.50, 0.50),
                pos=(-0.5, -1),
            ),
        ))
        
        cfgs.append(dict(
            name="obj2",
            obj_groups="bowl",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.cab,
                    top_size=(0.6, 0.4)
                ),
                size=(0.50, 0.50),
                pos=(0.5, -1),
            ),
        ))

        return cfgs    

    def _check_success(self):
        obj1_inside_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(self, "obj2", self.cab)    


        door_state = self.cab.get_door_state(env=self)
        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        return obj1_inside_cab and obj2_inside_cab and door_closed