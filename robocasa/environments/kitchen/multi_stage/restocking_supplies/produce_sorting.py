from robocasa.environments.kitchen.kitchen import *

class ProduceSorting(Kitchen):
    """
    Produce Sorting: composite task for the Restocking Supplies activity

    Simulates the task of putting items into the proper place in the fridge.
    
    Steps:
        Pick (fruit/veggie) from counter and place in corresponding shelf(1/2) in fridge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.produce = np.random.choice(['fruit', 'vegetable'])

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref(
            'fridge', dict(id=FixtureType.FRIDGE, full_name_check=True)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )

        self.init_robot_base_pos = self.fridge

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = (
            f"Move the {self.produce} to its correct location in the fridge."
        )
        return meta

    def _reset_internal(self):
        super()._reset_internal()
        self.fridge.open_door(env=self)
        self.produce = np.random.choice(['fruit', 'vegetable'])

    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=0),
                    size=(0.5, 0.4),
                    pos=(0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="fruit2",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=0),
                    size=(0.5, 0.4),
                    pos=(0, 0.7), 
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=-1),
                    size=(0.6, 0.4),
                    pos=(0, 1.0), 
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(rack_index=-1),
                    size=(0.6, 0.4),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        
        cfgs.append(
            dict(
                name='fruit',
                obj_groups='fruit',
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref="counter"),
                    size=(0.5, 0.40),
                    pos=(0.5, 0.5)
                ),
            )
        )

        cfgs.append(
            dict (
                name='vegetable',
                obj_groups='vegetable',
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref="counter"),
                    size=(0.5, 0.40),
                    pos=(1, 0.5)
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_name = self.objects[self.produce].name
        similar_name =self.objects['vegetable1'].name if self.produce=='vegetable' else self.objects['fruit1'].name

        obj_z = self.sim.data.body_xpos[self.obj_body_id[obj_name]][2]
        similar_z = self.sim.data.body_xpos[self.obj_body_id[similar_name]][2]

        same_shelf = abs(obj_z - similar_z)<0.02 
        in_fridge = OU.obj_inside_of(self, self.produce, self.fridge) 
        gripper_obj_far = OU.gripper_obj_far(self, obj_name=self.produce)

        return same_shelf and in_fridge and gripper_obj_far
