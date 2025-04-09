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

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        self.bottle_id = round(np.random.rand()) #0 is cond1, 1 is cond2
        self.obj_names = ['cond1_counter', 'cond2_counter', 'cond1_cab1', 'cond1_cab2', 'cond2_cab1', 'cond2_cab2']
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("cond1_counter")
        obj_name_2 = self.get_obj_lang("cond2_counter")

        ep_meta[
            "lang"
        ] = f"Pick one of ({obj_name_1}, {obj_name_2}), take the full bottle from the overhead cabinet, and place it on the counter."

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        self.bottle_id = round(np.random.rand())


    def _get_obj_cfgs(self):
        condiment_types = get_cats_by_type(
            types=["condiment"], obj_registries=self.obj_registries
        )
        cond1, cond2 = self.rng.choice(condiment_types, size=2, replace=False)
        cfgs = []
        cfgs.append(
            dict(
                name=self.obj_names[0],
                obj_groups=cond1,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(0.60, -1.0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[1],
                obj_groups=cond2,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(-0.60, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[2],
                obj_groups=cond1,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[3],
                obj_groups=cond1,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(0.33, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[4],
                obj_groups=cond2,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-0.33, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.obj_names[5],
                obj_groups=cond2,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = True
        for i in range(6):
            gripper_obj_far = gripper_obj_far and OU.gripper_obj_far(self, obj_name=self.obj_names[i])
                        
        condiment_on_counter = (OU.check_obj_fixture_contact(self, obj_name=self.obj_names[2*self.bottle_id + 2], fixture_name=self.counter)
                                or OU.check_obj_fixture_contact(self, obj_name=self.obj_names[2*self.bottle_id + 3], fixture_name=self.counter)
        )

        return gripper_obj_far and condiment_on_counter
