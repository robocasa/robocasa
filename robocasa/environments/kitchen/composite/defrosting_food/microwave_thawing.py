from robocasa.environments.kitchen.kitchen import *


class MicrowaveThawing(Kitchen):
    """
    Microwave Thawing: composite task for Defrosting Food activity.

    Simulates the task of defrosting food in a microwave.

    Steps:
        Pick the food from the counter and place it in the microwave.
        Then turn on the microwave.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave)
        )
        self.distr_counter = self.register_fixture_ref(
            "distractor_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.microwave,
            ),
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
        obj_name = self.get_obj_lang()
        ep_meta["lang"] = (
            f"Pick the {obj_name} from the counter and place it in the microwave. "
            "Then turn on the microwave."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="food",
                graspable=True,
                microwavable=True,
                freezable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
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
                    size=(0.50, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_in_microwave = OU.obj_inside_of(self, "obj", self.microwave)

        button_pressed = self.microwave.get_state()["turned_on"]
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_microwave and gripper_obj_far and button_pressed
