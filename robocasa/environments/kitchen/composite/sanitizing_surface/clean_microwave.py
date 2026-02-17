from robocasa.environments.kitchen.kitchen import *


class CleanMicrowave(Kitchen):
    """
    Clean Microwave: composite task for Sanitize Surface activity.

    Simulates the preparation for cleaning the microwave.

    Steps:
        Open the microwave, pick the sponge from the counter and place it in the
        microwave.
    """

    # Exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.microwave.close_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place the sponge inside the microwave to prepare for sanitizing."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="sponge",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.25, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_micro_contact = OU.obj_inside_of(self, "obj", self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self, th=0.10)
        return obj_micro_contact and gripper_obj_far
