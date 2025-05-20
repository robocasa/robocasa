from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type


class RefillCondiments(Kitchen):
    """
    Refill Condiments: composite task for Restocking Supplies activity.

    Simulates the task of refilling condiment bottles.

    Steps:
        Pick a random condiment bottle (of 2) that is 'running low', get the full
        one from the cabinet, place it next to the low bottle

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the 
            new condiment bottle is taken.
        
    """

    def __init__(self, *args, **kwargs):
        self.bottle_id = round(np.random.rand()) #0 is cond1, 1 is cond2
        self.obj_names = ['cond1_counter', 'cond2_counter', 'cond1_cab1', 'cond1_cab2', 'cond2_cab1', 'cond2_cab2']
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta[
            "lang"
        ] = f"Take a full {self.get_obj_lang(self.obj_names[self.bottle_id])} from the overhead cabinet, and place it on the counter next to the empty one."

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.bottle_id = round(np.random.rand())
        self.cab.open_door(min=1.0, max=1.0, env=self)


    def _get_obj_cfgs(self):
        cond_types = ['shaker', 'ketchup']
        cfgs = []
        cfgs.append(
            dict(
                name=self.obj_names[0],
                obj_groups=cond_types[0],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(0.60, 0.6),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[1],
                obj_groups=cond_types[1],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(-0.6, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[2],
                obj_groups=cond_types[0],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(1.0, 0.7),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[3],
                obj_groups=cond_types[0],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(0.5, 0.7),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[4],
                obj_groups=cond_types[1],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-0.5, 0.7),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[5],
                obj_groups=cond_types[1],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0, 0.7),
                ),
            )
        )


        
        return cfgs

    def _check_success(self):
        gripper_obj_far = all([OU.gripper_obj_far(self, obj_name=self.obj_names[i]) for i in range(6)])                        
        condiment_on_counter = (OU.check_obj_fixture_contact(self, obj_name=self.obj_names[2*self.bottle_id + 2], fixture_name=self.counter)
                                or OU.check_obj_fixture_contact(self, obj_name=self.obj_names[2*self.bottle_id + 3], fixture_name=self.counter)
        )

        return gripper_obj_far and condiment_on_counter
