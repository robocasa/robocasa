from robocasa.environments.kitchen.kitchen import *


class PrepareVeggiesForSteaming(Kitchen):
    """
    Prepare Veggies For Steaming: composite task for Steaming Vegetables activity.

    Simulates the task of rinsing vegetables and preparing them for steaming.

    Steps:
        1. Rinse the vegetables in the colander under running water for 50 timesteps
        2. Place the colander with vegetables next to the stove
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        # Choose a front burner for the saucepan
        valid_front_knobs = [
            k
            for (k, v) in self.stove.knob_joints.items()
            if v is not None and k.startswith("front")
        ]
        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            self.knob = self.rng.choice(valid_front_knobs)

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veggie1 = self.get_obj_lang("vegetable1")
        veggie2 = self.get_obj_lang("vegetable2")
        if veggie1 == veggie2:
            veggie_text = f"{veggie1}s"
        else:
            veggie_text = f"{veggie1} and {veggie2}"
        ep_meta["lang"] = (
            f"Rinse the {veggie_text} in the colander under running water briefly. "
            f"Then place the colander with the vegetables on the counter next to the stove to prepare for steaming."
        )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)
        self.washed_time = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="colander",
                obj_groups="colander",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        vegetable_options = [
            "broccoli",
            "carrot",
            "potato",
            "sweet_potato",
            "squash",
        ]

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=vegetable_options,
                placement=dict(
                    object="colander",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=vegetable_options,
                placement=dict(
                    object="colander",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan",
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veggie1_in_colander = OU.check_obj_in_receptacle(self, "vegetable1", "colander")
        veggie2_in_colander = OU.check_obj_in_receptacle(self, "vegetable2", "colander")

        if (
            self.sink.check_obj_under_water(self, "colander")
            and veggie1_in_colander
            and veggie2_in_colander
        ):
            self.washed_time += 1

        vegetables_washed = self.washed_time >= 50
        colander_on_counter = OU.check_obj_any_counter_contact(self, "colander")
        colander_near_stove = (
            OU.obj_fixture_bbox_min_dist(self, "colander", self.stove) < 0.3
        )
        gripper_far_colander = OU.gripper_obj_far(self, "colander")
        return (
            vegetables_washed
            and colander_on_counter
            and colander_near_stove
            and gripper_far_colander
        )
