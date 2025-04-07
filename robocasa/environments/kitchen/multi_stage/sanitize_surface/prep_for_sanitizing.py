from robocasa.environments.kitchen.kitchen import *


class PrepForSanitizing(Kitchen):
    """
    Prep For Sanitizing: composite task for Sanitize Surface activity.

    Simulates the preparation for sanitizing the surface.

    Steps:
        Pick the cleaners from the cabinet and place it on the counter.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            cleaners are picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
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
        obj1_name = self.get_obj_lang("obj1")
        obj2_name = self.get_obj_lang("obj2")
        ep_meta[
            "lang"
        ] = f"Pick the {obj1_name} and {obj2_name} from the cabinet and place them on the counter."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0.5, -1.0),
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
        obj1_on_counter = OU.check_obj_fixture_contact(self, "obj1", self.counter)
        obj2_on_counter = OU.check_obj_fixture_contact(self, "obj2", self.counter)
        gripper_obj_far = OU.gripper_obj_far(self, "obj1") and OU.gripper_obj_far(
            self, "obj2"
        )
        return obj1_on_counter and obj2_on_counter and gripper_obj_far
