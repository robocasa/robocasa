from robocasa.environments.kitchen.kitchen import *


class CandleCleanup(Kitchen):
    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)))
        self.init_robot_base_pos = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")
        ep_meta["lang"] = f"pick the {obj_name_1} and {obj_name_2} from the dining table and place it in the open cabinet"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="obj1",
            obj_groups="decoration",
            graspable=True,
            placement=dict(
                fixture=self.dining_table,
                size=(0.60, 0.30),
                sample_region_kwargs=dict(
                    ref=self.cab,
                ),
                pos=(0, -1),
                offset = (-0.05, 0)
            ),
        ))
        
        cfgs.append(dict(
            name="obj2",
            obj_groups="decoration",
            graspable=True,
            placement=dict(
                fixture=self.dining_table,
                sample_region_kwargs=dict(
                    ref=self.cab,
                ),
                size=(0.60, 0.30),
                pos=(0, -1),
                offset = (0.05, 0)
            ),
        ))

        cfgs.append(dict(
            name="distr_counter",
            obj_groups="all",
            placement=dict(
                fixture=self.dining_table,
                size=(1.0, 0.30),
                pos=(0.0, 0.0),
            ),
        ))

        

        cfgs.append(dict(
            name="distr_cab",
            obj_groups="all",
            placement=dict(
                fixture=self.cab,
                size=(1.0, 0.20),
                pos=(0.0, 1.0),
                offset=(0.0, 0.0),
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

        return door_closed and obj1_inside_cab and obj2_inside_cab


    



    