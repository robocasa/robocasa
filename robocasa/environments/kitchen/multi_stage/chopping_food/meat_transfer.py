from robocasa.environments.kitchen.kitchen import *


class MeatTransfer(Kitchen):
    def __init__(self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):    
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.5)))
        
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Retrieve a container (either a pan or a bowl) from the cabinet, " \
            "then place the raw meat into the container to avoid contamination."
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
    
    def _get_obj_cfgs(self):
        cfgs = []

        if random.random() < 0.5:
            cfgs.append(dict(
                name="container",
                obj_groups="pan",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.02),
                    pos=(0, 0),
                    rotation=(2*np.pi/8, 3*np.pi/8),
                )
            ))
        else:
            cfgs.append(dict(
                name="container",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    pos=(0, 0)
                )
            ))

        cfgs.append(dict(
            name="meat",
            obj_groups="meat",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.cab
                ),
                size=(0.5, 0.4),
                pos=(0.0, -1.0)
            ),
        ))
        return cfgs

    def _check_success(self):
        return OU.check_obj_fixture_contact(self, "container", self.counter) and \
            OU.gripper_obj_far(self, obj_name="meat") and \
            OU.check_obj_in_receptacle(self, "meat", "container")