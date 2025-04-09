from robocasa.environments.kitchen.kitchen import *

class ProduceSorting(Kitchen):
    """
    Produce Sorting: composite task for the Restocking Supplies activity

    Simulates the task of putting items into the proper place in the fridge.
    
    Steps:
        Pick (fruit/veggie) from counter and place in corresponding shelf(2/1) in fridge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_types = ['fruit', 'vegetable']
        self.obj_type_on_counter = round(np.random.rand())  # 0 for fruit, 1 for vegetable

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref(
            'fridge', dict(id='fridge')
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id="counter", size=(0.5, 0.5), ref=self.fridge)
        )

        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = (
            "Move the object on the counter (fruit/vegetable) to its correct location in the fridge."
        )
        return meta

    def _reset_internal(self):
        super()._reset_internal()
        
        self.fridge.set_door_state(min=0.9, max=1.0, env=self)
        self.obj_type_on_counter = round(np.random.rand())

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(ref="fridge_shelf2"),
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
                    sample_region_kwargs=dict(ref="fridge_shelf2"),
                    size=(0.5, 0.4),
                    pos=(-1.0, 1.0), 
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
                    sample_region_kwargs=dict(ref="fridge_shelf1"),
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
                    sample_region_kwargs=dict(ref="fridge_shelf1"),
                    size=(0.6, 0.4),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        obj_type_on_counter = self.obj_types[self.obj_type_on_counter] 
        cfgs.append(
            dict(
                name=f"{obj_type_on_counter}_on_counter",
                obj_groups=obj_type_on_counter,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref="counter"),
                    size=(0.5, 0.40),
                    pos=(0, 0, 0),  # Position for object on counter
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge",
                obj_groups="fridge",
                graspable=False,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(ref="fridge"),
                    size=(1.0, 1.0),
                    pos=(0, 0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        
        obj_on_counter_name = f"{self.obj_types[self.obj_type_on_counter]}_on_counter"
        target_shelf = "fridge_shelf2" if self.obj_types[self.obj_type_on_counter] == "fruit" else "fridge_shelf1"
    
        success = OU.obj_inside_of(self, obj_on_counter_name, self.fridge, region=target_shelf) and OU.gripper_obj_far(self, obj_name=obj_on_counter_name)
        return success
