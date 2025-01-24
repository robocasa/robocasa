from robocasa.environments.kitchen.kitchen import *


class RinseAndStackPlates(Kitchen):
    """
    Rinse and Stack Plates: composite task for Washing Dishes activity.

    Simulates the task of rinsing plates on the counter and stacking them in the sink.

    Steps:
        Move the robot to the stove-side counter to pick up a plate.
        Rinse the plate in the sink.
        Stack the rinsed plates in the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        self.cabinet_top = self.register_fixture_ref(
            "cabinet_top", dict(id=FixtureType.CABINET_TOP)
        )

        self.counter_non_corner = self.register_fixture_ref(
            "counter_non_corner", dict(id=FixtureType.COUNTER_NON_CORNER)
        )
        
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the plates from the counter, rinse them in the sink, and stack them in the sink."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        # Define the possible references
        possible_refs = [
            self.counter,
            self.stove,
            self.sink,
            self.cabinet_top,
            self.counter_non_corner,
        ]

        # Randomly select a reference for the plate placement
        chosen_ref1 = random.choice(possible_refs)
        chosen_ref2 = random.choice(possible_refs)

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=chosen_ref1,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=chosen_ref2,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        # omitted condition of sink faucet on, might add it later
        plate1_in_sink = OU.obj_inside_of(self, "plate1", self.sink)
        plate2_in_sink = OU.obj_inside_of(self, "plate2", self.sink)

        plate2_in_plate1 = OU.check_obj_in_receptacle(self, "plate2", "plate1")
        plate1_in_plate2 = OU.check_obj_in_receptacle(self, "plate1", "plate2")

        gripper_plate1_far = OU.gripper_obj_far(self, obj_name="plate1")
        gripper_plate2_far = OU.gripper_obj_far(self, obj_name="plate2")

        return (
            plate1_in_sink
            and plate2_in_sink
            and (plate2_in_plate1 or plate1_in_plate2)
            and gripper_plate1_far
            and gripper_plate2_far
        )