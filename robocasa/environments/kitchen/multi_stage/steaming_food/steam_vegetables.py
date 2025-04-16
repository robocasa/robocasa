from robocasa.environments.kitchen.kitchen import *


class SteamVegetables(Kitchen):
    """
    Steam Vegetables: composite task for Steaming Food activity.

    Simulates the task of steaming vegetables based on their cooking time.

    Steps:
        Place vegetables into the pot based on the amount of time it would take to
        steam each. e.g. potatoes and carrots would take the longest. Then, turn
        off the burner beneath the pot.

    Args:
        knob_id (str): The id of the knob who's burner the pot will be placed on.
            If "random", a random knob is chosen.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        # Used to see if the vegetables were placed in the wrong order. If so, task always fails.
        self.task_failed = False
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        # Pick a knob/burner on a stove and a counter close to it
        valid_knobs = [k for (k, v) in self.stove.knob_joints.items() if v is not None]
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.STOVE)
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Place vegetables into the pot based on the amount of time it would take to steam each, "
            "e.g. potatoes and carrots would take the longest. "
            "Then turn off the burner beneath the pot."
        )
        ep_meta["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="on", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable_hard",
                obj_groups=["potato", "carrot"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.50),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable_easy",
                obj_groups="vegetable",
                exclude_obj_groups=["potato", "carrot"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.50),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    size=(0.02, 0.02),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        if self.task_failed:
            return False

        # Must place vegetables into pot in sequence
        hard_in_pot = OU.check_obj_in_receptacle(self, "vegetable_hard", "pot")
        easy_in_pot = OU.check_obj_in_receptacle(self, "vegetable_easy", "pot")
        if easy_in_pot and not hard_in_pot:
            self.task_failed = True
            return False
        vegetables_in_pot = hard_in_pot and easy_in_pot

        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]
        knob_off = not (0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35)

        gripper_far = (
            OU.gripper_obj_far(self, "vegetable_hard")
            and OU.gripper_obj_far(self, "vegetable_easy")
            and OU.gripper_obj_far(self, "pot")
        )
        pot_on_stove = pan_on_stove = OU.check_obj_fixture_contact(
            self, "pot", self.stove
        )

        return knob_off and gripper_far and pot_on_stove and vegetables_in_pot
